import torch
from torch import nn
from transformers import ViTMAEForPreTraining
from .utils import get_2d_sincos_pos_embed, tokens_to_output
from hydra.utils import instantiate

class MAE(nn.Module):
    def __init__(
        self,
        checkpoint: str = "facebook/vit-mae-base",
        output: str = "dense",
        layer: int = -1,
        return_multilayer: bool = False,
        image_size: list[int] = (224, 224),
    ):
        """
        MAE Model initialization.

        Args:
            checkpoint: Path or name of the pretrained checkpoint.
            output: Output type, one of ["cls", "gap", "dense"].
            layer: Specific encoder layer to use. Defaults to -1 (last layer).
            return_multilayer: Whether to return outputs from multiple layers.
            image_size: The input image size as [height, width].
        """
        super().__init__()

        assert output in ["cls", "gap", "dense"], "Options: [cls, gap, dense]"
        
        self.name = 'MAE'
        self.output = output
        self.checkpoint_name = checkpoint.split("/")[-1]

        # Load the ViT model
        self.vit = ViTMAEForPreTraining.from_pretrained(checkpoint).vit
        self.vit = self.vit.eval()

        # Resize positional embedding
        self.patch_size = self.vit.config.patch_size
        self.image_size = image_size
        self.feat_h = self.image_size[0] // self.patch_size
        self.feat_w = self.image_size[1] // self.patch_size

        feat_dim = self.vit.config.hidden_size
        num_layers = len(self.vit.encoder.layer)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        # Determine feature dimension and layers to use
        if return_multilayer:
            self.dim = [feat_dim] * len(multilayers)
            self.multilayers = multilayers
        else:
            self.dim = feat_dim
            self.multilayers = [multilayers[-1] if layer == -1 else layer]

        # For logging purposes
        self.layer = "-".join(str(x) for x in self.multilayers)

    def resize_pos_embed(self, image_size):
        """
        Resize positional embeddings for new image size.

        Args:
            image_size: Tuple of (height, width).
        """
        assert image_size[0] % self.patch_size == 0
        assert image_size[1] % self.patch_size == 0

        self.feat_h = image_size[0] // self.patch_size
        self.feat_w = image_size[1] // self.patch_size

        embed_dim = self.vit.config.hidden_size
        self.vit.embeddings.patch_embeddings.image_size = image_size

        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.feat_h, self.feat_w), add_cls_token=True
        )
        device = self.vit.embeddings.patch_embeddings.projection.weight.device
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0).to(device=device),
            requires_grad=False,
        )

    def embed_forward(self, embedder, pixel_values):
        """
        Forward pass through the embedding layer.

        Args:
            embedder: Embedding layer.
            pixel_values: Input image tensor.

        Returns:
            Patch embeddings with positional encodings.
        """
        embeddings = embedder.patch_embeddings(pixel_values)
        embeddings = embeddings + embedder.position_embeddings[:, 1:, :]

        cls_token = embedder.cls_token + embedder.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        return embeddings

    def forward(self, images):
        """
        Forward pass through the model.

        Args:
            images: Input image tensor of shape (B, C, H, W).

        Returns:
            Features from the specified layers.
        """
        if self.image_size != images.shape[-2:]:
            self.resize_pos_embed(images.shape[-2:])

        head_mask = self.vit.get_head_mask(None, self.vit.config.num_hidden_layers)
        embedding_output = self.embed_forward(self.vit.embeddings, images)

        encoder_outputs = self.vit.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=self.vit.config.output_attentions,
            output_hidden_states=True,
            return_dict=self.vit.config.return_dict,
        )

        outputs = []
        for layer_i in self.multilayers:
            x_i = encoder_outputs.hidden_states[layer_i]
            x_i = tokens_to_output(
                self.output, x_i[:, 1:], x_i[:, 0], (self.feat_h, self.feat_w)
            )
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
