import torch
import torch.nn as nn
import math
from torch.nn.init import trunc_normal_
from .utils import (
    Block,
    CrossAttention,
    CrossAttentionBlock
)


class AttentivePooler(nn.Module):
    """
    AttentivePooler can be configured for EITHER:
      - A single query for entire image (num_queries=1), i.e. global pooling
      - One query per pixel (num_queries=H*W), i.e. pixel-wise features
      - Or anything in between
    """
    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        # A 'complete' cross-attention block vs. a simpler cross-attn-only module
        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer
            )
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias
            )

        # Additional self-attention blocks (depth-1 of them)
        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=False,
                    norm_layer=norm_layer
                )
                for _ in range(depth - 1)
            ])

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        """Initialize weights (truncated normal for Linear & Conv2d, constants for bias/norm)."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        """
        Some implementations rescale weights by sqrt(2.0 * layer_id).
        This helps with stable training for deep networks.
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        # Cross-attention block
        if self.complete_block:
            # The CrossAttentionBlock has:
            #   self.xattn.proj.weight
            #   self.mlp.fc2.weight  (or self.mlp's second Linear)
            # Adjust accordingly based on your MLP/attention structure
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            # The MLP is: MLP->fc1, fc2
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)

        # Additional blocks
        if self.blocks is not None:
            for layer_id, blk in enumerate(self.blocks, 1):
                # Each Block has: self.attn.proj, self.mlp.fc2, etc.
                # Here we need to locate them accordingly:
                #   blk.attn.proj -> for 'Attention' class
                #   blk.mlp.fc2   -> for 'MLP' class
                # But in your code, Block uses MultiheadAttention or the user-supplied 'Attention' class.
                # Adjust the references as needed:
                attn_proj = blk.attn.proj if hasattr(blk.attn, 'proj') else blk.attn.proj
                mlp_fc2   = blk.mlp.fc2
                rescale(attn_proj.weight.data, layer_id + 1)
                rescale(mlp_fc2.weight.data, layer_id + 1)

    def forward(self, x):
        """
        x: [B, N, embed_dim]
            - B = batch size
            - N = sequence length (e.g. flattened tokens for an image)
            - embed_dim = feature dimension
        Returns: [B, num_queries, embed_dim]
        """
        B = x.size(0)
        # 1) Repeat the query tokens to match batch size
        q = self.query_tokens.repeat(B, 1, 1)  # [B, num_queries, embed_dim]

        # 2) Cross-attention
        q = self.cross_attention_block(q, x)   # [B, num_queries, embed_dim]

        # 3) Additional self-attention blocks (optional)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)  # [B, num_queries, embed_dim]

        return q


class AttentiveProbe(nn.Module):
    """
    Attentive Probe for pixel-wise classification using AttentivePooler.

    Key difference from typical classification:
      - We set num_queries = (H * W).
      - After cross-attention, we get a feature for each "pixel query".
      - A final linear layer produces [B, H*W, num_classes].
      - Reshape to [B, num_classes, H, W].
    """

    def __init__(
            self,
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            depth=1,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            qkv_bias=True,
            num_classes=3,
            device="cuda",
            complete_block=True,
            image_height=224,
            image_width=416,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.device = device
        self.name = "attentive_probe"

        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.qkv_bias = qkv_bias
        self.norm_layer = norm_layer
        self.init_std = init_std
        self.complete_block = complete_block
        self.device = device

        self.image_height = image_height
        self.image_width = image_width
        self.num_queries = image_height * image_width  # one query per pixel

        self.pooler = AttentivePooler(
            num_queries=self.num_queries,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        ).to(device)

        # A linear layer to map from [embed_dim] -> [num_classes]
        self.linear = nn.Linear(embed_dim, num_classes, bias=True).to(device)

    def update_n_classes(self, num_classes):
        self.num_classes = num_classes
        self.linear = nn.Linear(self.embed_dim, self.num_classes, bias=True).to(self.device)


    def update_input_dim(self, input_dim):
        self.embed_dim = input_dim
        self.pooler = AttentivePooler(
            num_queries=self.num_queries,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            depth=self.depth,
            norm_layer=self.norm_layer,
            init_std=self.init_std,
            qkv_bias=self.qkv_bias,
            complete_block=self.complete_block,
        ).to(self.device)

        self.linear = nn.Linear(self.embed_dim, self.num_classes, bias=True).to(self.device)

    def update_image_dim(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.num_queries = image_height * image_width

        self.pooler = AttentivePooler(
            num_queries=self.num_queries,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            depth=self.depth,
            norm_layer=self.norm_layer,
            init_std=self.init_std,
            qkv_bias=self.qkv_bias,
            complete_block=self.complete_block,
        ).to(self.device)


    def forward(self, x):
        """
        x: [B, C, H, W]
            - B = batch size
            - C = embed_dim (channel dimension that matches self.embed_dim)
            - H, W = image spatial dimensions

        Returns: [B, num_classes, H, W]
        """
        print("pooler input shape:", x.shape)
        B, C, H, W = x.shape
        if H != self.image_height or W != self.image_width:
            print("Image spatial dimensions:", H, W)
            print("Adapt pooler")
            self.update_image_dim(image_height=H, image_width=W)

        if C != self.embed_dim:
            raise ValueError(
                f"Expected channel dim = {self.embed_dim}, but got {C}."
            )
        if H != self.image_height or W != self.image_width:
            raise ValueError(
                f"Input size ({H}x{W}) does not match expected size "
                f"({self.image_height}x{self.image_width})."
            )

        # 1) Flatten spatial dims: [B, C, H, W] -> [B, H*W, C]
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        print("after reshape:", x.shape)

        # 2) Attentive pooling: produces [B, H*W, embed_dim]
        x = self.pooler(x)
        print("after pooler:", x.shape)

        # 3) Linear classification on each query => [B, H*W, num_classes]
        x = self.linear(x)
        print("after linear:", x.shape)

        # 4) Reshape back to [B, num_classes, H, W]
        x = x.reshape(B, H, W, self.num_classes).permute(0, 3, 1, 2)
        print("final output shape:", x.shape)

        return x