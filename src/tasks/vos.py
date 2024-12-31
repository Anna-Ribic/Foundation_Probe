# src/tasks/vos.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
from PIL import Image
import numpy as np
from tqdm import tqdm
import queue
from hydra.utils import instantiate
from urllib.request import urlopen
from src.tasks.utils import (
    imwrite_indexed,
    mask_overlay,
    ResultWriter,
    norm_mask,
    restrict_neighborhood,
    mask2tensor
)
from hydra.core.hydra_config import HydraConfig


#TODO move somewhere
resize = lambda x, new_size: F.interpolate(x, size=new_size, mode='bilinear', align_corners=False,
                                               recompute_scale_factor=False)

class VOSTask:
    def __init__(
        self,
        result_path: str,
        eval_size: list,
        feat_res: list,
        use_knn: bool,
        first_frame_iter: int,
        crop: bool,
        loss: str,
        downsample_factor: int,
        knn_parameters: dict,
    ):
        """
        Initialize the VOS Task.

        Args:
            model (nn.Module): The backbone model.
            probe (nn.Module): The probe (e.g., linear head, MLP).
            dataset: The dataset to evaluate on.
            result_path (str): Path to save results.
            eval_size (list): Evaluation image size [H, W].
            n_iter (int): Number of iterations for probe training.
            linprob (bool): Whether to use linear probing.
            multiple (bool): Whether to use multiple probes.
            loss (str): Loss function name.
        """
        self.name = "Video Object Segmentation"

        self.base_path = result_path
        self.eval_size_init = eval_size
        self.n_iter = first_frame_iter
        self.loss_f = getattr(F, loss)
        self.downsample_factor = downsample_factor
        self.crop = crop
        self.use_knn = use_knn
        self.feat_res = feat_res
        self.knn_parameters = knn_parameters if knn_parameters else {}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('useknnd', use_knn)
        if self.use_knn:
            print('Use KNN for label propagation: PROBE NOT IN USE!')

    def evaluate(self, model: nn.Module, probe: nn.Module, dataset: torch.utils.data.Dataset):
        """
        Run evaluation on the dataset.
        """

        result_path = os.path.join(self.base_path, f'{model.config_name}', probe.name if not self.use_knn else 'KNN', 'iter_'+str(self.n_iter) if not self.use_knn else '', 'eval_size_'+str(self.eval_size_init), 'no_crop' if not self.crop else 'crop')

        model.to(self.device)
        probe.to(self.device)

        if self.use_knn:
            print('Evaluating with KNN')
            result_path = os.path.join(
                str(result_path),
                f"down_factor_{self.downsample_factor}" if self.feat_res is None else f"feat_res_{self.feat_res}",
                f"n_frames_{self.knn_parameters.get('n_last_frames', 5)}",
                f"topk_{self.knn_parameters.get('topk', 3)}",
                f"neigh_size_{self.knn_parameters.get('size_mask_neighborhood', 7)}"
            )
            print(f'Storing results in {result_path}')
            self.run_validation_label_prop(model, dataset, self.knn_parameters, result_path=result_path, eval_size_init=self.eval_size_init,
                                      feat_res_init=self.feat_res, downsample_factor_feat=self.downsample_factor)

        else:
            print(f'Storing results in {result_path}')
            self.run_validation_mlp(model, probe, dataset, loss_f=self.loss_f,
                               result_path=result_path, eval_size_init=self.eval_size_init, n_iter=self.n_iter)



    def train(self):
        pass

    def run_mlp_once(self, result_path, video, net, probe, loss_f, eval_size, color_palette, n_iter=500):

        denorm = tf.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                              std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

        tf_val = tf.Resize(eval_size)
        tf_val_mask = tf.Resize(eval_size)
        tf_t = tf.ToTensor()
        tf_val_net = tf.Compose([tf.ToTensor(),
                                 tf.Normalize(mean=[0.485, 0.456, 0.406], \
                                              std=[0.229, 0.224, 0.225])])

        parts = video['images'][0].split("/")
        name = parts[-2]

        video_folder = os.path.join(result_path, name)
        os.makedirs(video_folder, exist_ok=True)

        images = []
        masks_gt = []
        images_un = []
        masks_gt_up = []
        for n in range(video["len"]):
            image_pil = Image.open(video["images"][n]).convert('RGB')
            images_un.append(tf_t(image_pil))
            image_pil = tf_val(image_pil)

            _, ori_h, ori_w = tf_t(Image.open(video["masks"][n])).shape
            mask = tf_val_mask(Image.open(video["masks"][n]))
            mask = torch.from_numpy(np.array(mask, np.int64, copy=False))
            masks_gt.append(mask)

            mask_up = Image.open(video["masks"][n])
            mask_up = torch.from_numpy(np.array(mask_up, np.int64, copy=False))
            masks_gt_up.append(mask_up)

            im = tf_val_net(image_pil)
            images.append(im)

        num_cls = masks_gt_up[0].max() + 1

        # initialising the mask
        mask_init = mask2tensor(masks_gt[0], num_cls)[None, ...].cuda()
        mask_init_orig = mask2tensor(masks_gt_up[0], num_cls)[None, ...].cuda()

        frame_nm = video["images"][0].split('/')[-1].replace(".jpg", ".png")
        imwrite_indexed(os.path.join(video_folder, frame_nm), masks_gt_up[0].numpy().astype(np.uint8), color_palette)

        probe.update_n_classes(num_cls)
        mlp_cls = probe

        cls_optim = torch.optim.Adam(mlp_cls.parameters(), weight_decay=0.0005, lr=0.005)

        # fitting a classifier to the first image
        with torch.no_grad():
            x = images[0][None, ...].cuda()

            print('Extracting Features')

            features = net(x)
            features = resize(features, eval_size)

        pbar = tqdm(range(n_iter))
        for itr in pbar:
            cls_optim.zero_grad()

            logits = mlp_cls(features)

            logits_up = resize(logits, (ori_h, ori_w))
            loss = loss_f(logits_up, mask_init_orig)

            loss.backward()
            cls_optim.step()

            pbar.set_description("Loss = {:4.3f}".format(loss.item()), refresh=True)

        for n, im in enumerate(images[1:]):

            x = im[None, ...].cuda()

            with torch.no_grad():
                features = net(x)
                features = resize(features, eval_size)

                logits = mlp_cls(features)
                #
                logits_up = resize(logits, (ori_h, ori_w))
                mask_idx = logits_up.argmax(1)

                frame_nm = video["images"][n + 1].split('/')[-1].replace(".jpg", ".png")
                imwrite_indexed(os.path.join(video_folder, frame_nm), \
                                mask_idx[0].cpu().numpy().astype(np.uint8), color_palette)

    def run_validation_mlp(self, net, probe, dataset, loss_f=F.cross_entropy, result_path='./eval_results', eval_size_init=None,n_iter=500):

        color_palette = []
        for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
            color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
        color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1, 3)

        tf_t = tf.ToTensor()

        image_folder = os.path.join(result_path, 'vis')
        os.makedirs(image_folder, exist_ok=True)

        for video in dataset.videos:
            print('evalinit', eval_size_init)
            if len(eval_size_init) > 1:
                eval_size = eval_size_init
                print(f"Fixed eval size {eval_size}")
            else:

                img_size = eval_size_init[0]
                # img_size = 576

                _, ori_h, ori_w = tf_t(Image.open(video["masks"][0])).shape

                print(f"Original H: {ori_h}, Original W: {ori_w}")

                ratio = ori_w / ori_h
                print(f"Ratio: {ratio}")

                print('imgsize', img_size)
                eval_h = img_size
                eval_w = int((img_size * ratio) // 16 * 16)

                print(f"Eval H: {eval_h}, Eval W: {eval_w}")

                eval_size = (eval_h, eval_w)

                ##FIX in case feat size not even number
                print(eval_size[1], eval_size[1] // 16)
                if (eval_size[1] // 16) % 2 != 0:
                    print("Make Feat w to even number")
                    eval_size = (eval_size[0], eval_size[1] + 16)
                    print(f"New Eval H: {eval_size[0]}, Eval W: {eval_size[1]}")

            self.run_mlp_once(result_path, video, net, probe, loss_f, eval_size, color_palette, n_iter=n_iter)


    def run_validation_label_prop(self, net, dataset, args, result_path='./eval_results', eval_size_init=None,
                                  feat_res_init=None, downsample_factor_feat=8):

        tf_t = tf.ToTensor()

        with torch.no_grad():
            def mask2tensor(mask, num_cls=3):
                h, w = mask.shape
                ones = torch.ones(1, h, w)
                zeros = torch.zeros(num_cls, h, w)

                # assert mask.max() < num_cls, "{} >= {}".format(mask.max(), num_cls)
                return zeros.scatter(0, mask[None, ...], ones)

            # video = dataset.videos[1]
            color_palette = []
            for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
                color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
            color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1, 3)

            image_folder = os.path.join(result_path, 'vis')
            os.makedirs(image_folder, exist_ok=True)

            for video in dataset.videos:

                parts = video['images'][0].split("/")
                name = parts[-2]

                if len(eval_size_init) > 1:
                    eval_size = eval_size_init
                    print(f"Fixed eval size {eval_size}")

                else:

                    _, ori_h, ori_w = tf_t(Image.open(video["masks"][0])).shape

                    print(f"Original H: {ori_h}, Original W: {ori_w}")

                    ratio = ori_w / ori_h
                    print(f"Ratio: {ratio}")

                    img_size = eval_size_init[0]
                    eval_h = img_size

                    print(img_size)
                    eval_w = int((img_size * ratio) // 16 * 16)

                    print(f"Eval H: {eval_h}, Eval W: {eval_w}")

                    eval_size = (eval_h, eval_w)

                if len(feat_res_init) > 1:
                    print(f"Fixed feat resolution:")
                    feat_res = feat_res_init
                elif feat_res_init is not None:
                    assert ratio is not None
                    print(f"Fixed feat h: ", feat_res_init)
                    feat_res = feat_res_init[0], int(feat_res_init[0] * ratio)

                else:
                    print(f'Feature resolution downsampled eval size by {downsample_factor_feat}')
                    feat_res = eval_size[0] // downsample_factor_feat, eval_size[1] // downsample_factor_feat

                print(f"Feat H: {feat_res[0]}, Feat W: {feat_res[1]}")

                print(eval_size[1], eval_size[1] // 16)
                if (eval_size[1] // 16) % 2 != 0:
                    print("Make Feat w to even number")
                    eval_size = (eval_size[0], eval_size[1] + 16)
                    feat_res = (feat_res[0], feat_res[1] + 1)
                    print(f"New Eval H: {eval_size[0]}, Eval W: {eval_size[1]}")
                    print(f"New Feat H: {feat_res[0]}, Feat W: {feat_res[1]}")

                """##FIX in case feat size not even number
                if feat_res[1] % 2 != 0:
                    print("Make Feat w to even number")
                    feat_res = (feat_res[0], feat_res[1] + 1)
                    eval_size =(eval_size[0], eval_size[1] + 16)
                    print(f"New Eval H: {eval_size[0]}, Eval W: {eval_size[1]}")
                    print(f"New Feat H: {feat_res[0]}, Feat W: {feat_res[1]}")"""

                tf_val = tf.Resize(eval_size)  # tf.Compose([tf.Resize(224), tf.CenterCrop(224)])
                tf_val_mask = tf.Resize(
                    feat_res)  # tf.Resize((eval_size[0]//net.patch_size, eval_size[1] // net.patch_size)) #tf.Resize((60, 106))#tf.Resize((30, 53))  # tf.Compose([tf.Resize(224), tf.CenterCrop(224)])
                tf_val_net = tf.Compose([tf.ToTensor(),
                                         tf.Normalize(mean=[0.485, 0.456, 0.406], \
                                                      std=[0.229, 0.224, 0.225])])

                video_folder = os.path.join(result_path, name)
                os.makedirs(video_folder, exist_ok=True)

                images = []
                masks_gt = []
                masks_gt_up = []
                images_un = []
                for n in range(video["len"]):
                    image_pil = Image.open(video["images"][n]).convert('RGB')
                    images_un.append(tf_t(image_pil))
                    image_pil = tf_val(image_pil)
                    _, ori_h, ori_w = tf_t(Image.open(video["masks"][n])).shape

                    mask = tf_val_mask(Image.open(video["masks"][n]))
                    mask = torch.from_numpy(np.array(mask, np.int64, copy=False))
                    masks_gt.append(mask)

                    mask_up = Image.open(video["masks"][n])
                    mask_up = torch.from_numpy(np.array(mask_up, np.int64, copy=False))
                    masks_gt_up.append(mask_up)

                    im = tf_val_net(image_pil)
                    images.append(im)

                num_classes = masks_gt_up[0].max() + 1

                first_seg = mask2tensor(masks_gt[0], num_cls=num_classes)[None, ...].cuda()

                fseg = F.interpolate(first_seg, size=eval_size, mode='bilinear', align_corners=False,
                                     recompute_scale_factor=False)

                mask_est_d = mask2tensor(masks_gt_up[0], num_cls=num_classes)[None, ...].cuda()
                mask_est_d = F.interpolate(mask_est_d, size=(ori_h, ori_w), mode='bilinear', align_corners=False,
                                           recompute_scale_factor=False)
                mask_est_d = norm_mask(mask_est_d.squeeze())
                _, mask_est_d = torch.max(mask_est_d, dim=0)

                mask_est_d = np.array(mask_est_d.squeeze().cpu(), dtype=np.uint8)
                mask_est_d = np.array(Image.fromarray(mask_est_d).resize((ori_w, ori_h), 0))

                frame_nm = video["images"][0].split('/')[-1].replace(".jpg", ".png")
                imwrite_indexed(os.path.join(video_folder, frame_nm), mask_est_d, color_palette)

                frame1 = images[0]
                frame1 = frame1[None, ...].cuda()

                # extract first frame feature
                # rendered_rgb, _, frame1_feat, regs, render_params = net.forward_and_render(frame1)
                frame1_feat = net(frame1)
                frame1_feat = F.interpolate(frame1_feat, size=feat_res, mode='bilinear', align_corners=False,
                                            recompute_scale_factor=False)

                frame1_feat = frame1_feat.squeeze().view(net.dim, -1)

                que = queue.Queue(7)

                mask_neighborhood = None
                for cnt in tqdm(range(1, video['len'])):
                    frame_tar = images[cnt]  # read_frame(frame_list[cnt])[0]
                    # outs["input"].append(frame_tar.unsqueeze(0).cpu())

                    # we use the first segmentation and the n previous ones
                    used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
                    used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

                    frame_tar_avg, feat_tar, mask_neighborhood = self.label_propagation(net, args, frame_tar,
                                                                                   used_frame_feats,
                                                                                   used_segs, mask_neighborhood,
                                                                                   feat_res=feat_res)

                    # pop out oldest frame if neccessary
                    if que.qsize() == 7:
                        que.get()
                    # push current results into queue
                    seg = frame_tar_avg.detach().clone()  # copy.deepcopy(frame_tar_avg)
                    que.put([feat_tar, seg])

                    # upsampling & argmax
                    frame_tar_avg = F.interpolate(frame_tar_avg, size=eval_size, mode='bilinear', align_corners=False,
                                                  recompute_scale_factor=False)[0]

                    frame_tar_avg = norm_mask(frame_tar_avg)
                    # _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

                    _, mask_est_d = torch.max(frame_tar_avg, dim=0)

                    mask_est_d = np.array(mask_est_d.squeeze().cpu(), dtype=np.uint8)
                    mask_est_d = np.array(Image.fromarray(mask_est_d).resize((ori_w, ori_h), 0))

                    frame_nm = video["images"][cnt].split('/')[-1].replace(".jpg", ".png")
                    imwrite_indexed(os.path.join(video_folder, frame_nm), mask_est_d, color_palette)



    def label_propagation(self, model, args, frame_tar, list_frame_feats, list_segs, mask_neighborhood=None,
                          feat_res=(30, 53)):
        """
        propagate segs of frames in list_frames to frame_tar
        """

        ## we only need to extract feature of the target frame
        # rendered_rgb, _, feat_tar, regs, render_params = net.forward_and_render(frame_tar.unsqueeze(0).cuda())
        feat_tar = model(frame_tar.unsqueeze(0).cuda())
        feat_tar = F.interpolate(feat_tar, size=feat_res, mode='bilinear', align_corners=False,
                                 recompute_scale_factor=False)

        h, w = feat_tar.shape[-2:]
        feat_tar = feat_tar.squeeze().view(model.dim, -1).squeeze()

        return_feat_tar = feat_tar  # dim x h*w

        ncontext = len(list_frame_feats)
        feat_sources = torch.stack(list_frame_feats)  # nmb_context x dim x h*w

        feat_tar = F.normalize(feat_tar, dim=1, p=2)
        feat_sources = F.normalize(feat_sources, dim=1, p=2)

        feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1).movedim(1, -1)
        aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1)  # nmb_context x h*w (tar: query) x h*w (source: keys)

        if args.size_mask_neighborhood > 0:
            if mask_neighborhood is None:
                mask_neighborhood = restrict_neighborhood(h, w, args.size_mask_neighborhood)
                mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
            aff *= mask_neighborhood

        aff = aff.transpose(2, 1).reshape(-1, h * w)  # nmb_context*h*w (source: keys) x h*w (tar: queries)
        tk_val, _ = torch.topk(aff, dim=0, k=args.topk)
        tk_val_min, _ = torch.min(tk_val, dim=0)
        aff[aff < tk_val_min] = 0

        aff = aff / torch.sum(aff, keepdim=True, axis=0)

        list_segs = [s.cuda() for s in list_segs]
        segs = torch.cat(list_segs)
        nmb_context, C, h, w = segs.shape
        segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T  # C x nmb_context*h*w
        seg_tar = torch.mm(segs, aff)
        seg_tar = seg_tar.reshape(1, C, h, w)
        return seg_tar, return_feat_tar, mask_neighborhood






