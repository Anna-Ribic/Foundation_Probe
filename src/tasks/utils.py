import torch
import numpy as np
from PIL import Image



def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if (mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt / mask_cnt.max()
            mask[cnt, :, :] = mask_cnt
    return mask

def restrict_neighborhood(h, w, size_mask_neighborhood):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)

def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')



def mask2rgb(mask, palette):
    mask_rgb = palette(mask)
    mask_rgb = mask_rgb[:, :, :3]
    return mask_rgb


# Function to create an overlay of the mask on the image
def mask_overlay(mask, image, palette):
    """Creates an overlayed mask visualization"""
    mask_rgb = mask2rgb(mask, palette)
    return 0.3 * image + 0.7 * mask_rgb


# Class to handle saving results
class ResultWriter:

    def __init__(self, palette, out_path):
        self.palette = palette
        self.out_path = out_path

    def save(self, frames, masks_pred, masks_gt, seq_name):
        subdir_vis = os.path.join(self.out_path, "{}_vis".format(seq_name))
        os.makedirs(subdir_vis, exist_ok=True)

        for frame_id in range(frames.shape[0]):
            frame = frames[frame_id].numpy()
            frame = np.transpose(frame, [1, 2, 0])

            mask_pred = masks_pred[frame_id].numpy().astype(np.uint8)
            mask_gt = masks_gt[frame_id].numpy().astype(np.uint8)

            overlay_pred = mask_overlay(mask_pred, frame, self.palette)
            overlay_gt = mask_overlay(mask_gt, frame, self.palette)

            combined_overlay = np.concatenate((overlay_pred, overlay_gt), axis=1)

            filepath = os.path.join(subdir_vis, "{}.png".format(frame_id))
            image_pil = tf.ToPILImage()((combined_overlay*255).astype(np.uint8))
            image_pil.save(filepath)

def mask2tensor(mask, num_cls=3):
    h, w = mask.shape
    ones = torch.ones(1, h, w)
    zeros = torch.zeros(num_cls, h, w)

    #assert mask.max() < num_cls, "{} >= {}".format(mask.max(), num_cls)
    return zeros.scatter(0, mask[None, ...], ones)