import torch

from .utils import center_padding, tokens_to_output


class DINO(torch.nn.Module):
    def __init__(
        self,
        config_name,
        checkpoint='dino_vitb16',
        output="dense",
        layer=-1,
        return_multilayer=False,
        image_size: list[int] = (224, 224)
    ):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }

        # get model
        self.config_name = config_name
        self.checkpoint_name = checkpoint
        self.name, self.model_name = checkpoint.split("_")
        print("dino_name", self.name, self.model_name)
        dino_vit = torch.hub.load(f"facebookresearch/{self.name}", self.checkpoint_name)
        self.vit = dino_vit.eval().to(torch.float32)
        self.has_registers = "_reg" in self.checkpoint_name
        self.image_size = image_size


        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[self.model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):

        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if "dinov2" in self.checkpoint_name:
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
