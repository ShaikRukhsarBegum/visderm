"""
VisDerm: Split-ViT model.

Partitions DeiT-Tiny at transformer block 6, exposing client_forward and
server_forward methods so that an STP-DP (or any other) module can be
inserted between them at inference time.

Reference: Section 2.2 of the paper.
"""
import torch
import torch.nn as nn
import timm


class SplitViT(nn.Module):
    """
    Split Vision Transformer with DeiT-Tiny backbone.

    The network is partitioned at transformer block ``split_point``:
      * Client side runs patch embedding, positional encoding, and
        blocks ``[1, ..., split_point]``.
      * Server side runs blocks ``[split_point+1, ..., 12]``, layer norm,
        and the classification head.

    The intermediate representation (shape ``[B, 197, 192]`` at split_point=6)
    is what gets transmitted in the deployment scenario, optionally after
    being passed through a privacy mechanism (see ``src/privacy.py``).

    Args:
        num_classes: Number of dx classes (default 7 for HAM10000).
        split_point: Index of the transformer block at which to split
            (default 6 — empirically validated as the optimal balance
            between client/server computation and payload size).
        pretrained: Whether to initialize from ImageNet-pretrained DeiT-Tiny
            weights via ``timm`` (default True).
    """

    def __init__(self, num_classes: int = 7, split_point: int = 6, pretrained: bool = True):
        super().__init__()
        full_model = timm.create_model(
            'deit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.patch_embed = full_model.patch_embed
        self.cls_token = full_model.cls_token
        self.pos_embed = full_model.pos_embed
        self.pos_drop = full_model.pos_drop
        self.client_blocks = nn.Sequential(*list(full_model.blocks.children())[:split_point])
        self.server_blocks = nn.Sequential(*list(full_model.blocks.children())[split_point:])
        self.norm = full_model.norm
        self.head = full_model.head
        self.embed_dim = full_model.embed_dim
        self.num_heads = full_model.blocks[0].attn.num_heads
        self.split_point = split_point

    def client_forward(self, x: torch.Tensor):
        """
        Run patch embedding + first ``split_point`` transformer blocks.

        Returns:
            features: ``[B, 197, 192]`` — intermediate token representation.
            spatial_attn: ``[B, 14, 14]`` — class-token attention map at the
                final client-side block, useful for explainability and for
                attention-aware noise allocation (LSP-DP).
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.client_blocks(x)

        # Compute class-token spatial attention at the final client block,
        # used for explainability figures and LSP-DP comparison.
        attn_layer = self.client_blocks[-1].attn
        B, N, C = x.shape
        qkv = attn_layer.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        aw = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        aw = aw.softmax(dim=-1)
        spatial_attn = aw[:, :, 0, 1:].mean(dim=1)
        H = int(spatial_attn.shape[-1] ** 0.5)
        return x, spatial_attn.reshape(B, H, H)

    def server_forward(self, features: torch.Tensor):
        """
        Run the remaining transformer blocks, layer norm, and classifier head.

        Args:
            features: Either the unmodified output of ``client_forward``, or
                the output of a privacy mechanism applied to that output.
                Shape ``[B, N, 192]`` where ``N`` may be less than 197 if
                tokens were pruned (e.g., STP-DP).
        """
        x = self.server_blocks(features)
        x = self.norm(x)
        return self.head(x[:, 0])

    def forward(self, images: torch.Tensor):
        """End-to-end forward pass without privacy mechanism."""
        features, attention = self.client_forward(images)
        return self.server_forward(features), attention
