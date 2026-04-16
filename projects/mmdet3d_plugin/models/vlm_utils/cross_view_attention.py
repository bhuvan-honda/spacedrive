"""
Cross-View 3D-Aware Attention (3D-XVA)

Uses 3D positional encodings to enable attention between vision tokens from
different camera views that correspond to nearby 3D locations. This creates
spatially consistent multi-view features before they enter the VLM.

The module operates on PE embeddings (B, N_views * H_tok * W_tok, hidden_dim)
and their corresponding 3D coordinates (B, N_views * H_tok * W_tok, 3).
"""

import torch
import torch.nn as nn


class CrossView3DAttention(nn.Module):
    """
    Cross-View 3D-Aware Attention module.

    For each token, attends to tokens from OTHER camera views that are
    within a 3D distance threshold, using the PE-augmented features as
    queries/keys/values.

    Args:
        hidden_dim (int): Dimension of the input embeddings (must match llm_hidden_dim).
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers for cross-view attention.
        num_views (int): Number of camera views (default: 6 for nuScenes).
        distance_threshold (float): 3D distance threshold in meters for sparse attention.
            Tokens farther than this won't attend to each other.
            If <= 0, full cross-view attention is used (no distance masking).
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads=8,
        num_layers=2,
        num_views=6,
        distance_threshold=10.0,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_views = num_views
        self.distance_threshold = distance_threshold
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.layers = nn.ModuleList([
            CrossView3DAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, pos_embed, coords3d):
        """
        Args:
            pos_embed: (B, N_views * H * W, hidden_dim) — PE embeddings for all views
            coords3d: (B, N_views * H * W, 3) — 3D world coordinates for all tokens

        Returns:
            pos_embed_out: (B, N_views * H * W, hidden_dim) — cross-view refined PE embeddings
        """
        B, total_tokens, C = pos_embed.shape
        tokens_per_view = total_tokens // self.num_views

        # Clamp coords to avoid NaN/Inf from bad depth predictions
        coords3d = coords3d.clone()
        coords3d = torch.nan_to_num(coords3d, nan=0.0, posinf=100.0, neginf=-100.0)

        # Build cross-view attention mask based on 3D distance
        # attn_mask: (B, total_tokens, total_tokens), True = MASKED (cannot attend)
        attn_mask = self._build_cross_view_mask(coords3d, tokens_per_view, B)

        x = pos_embed
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.norm(x)
        return x

    def _build_cross_view_mask(self, coords3d, tokens_per_view, B):
        """
        Build an attention mask that:
        1. Blocks self-view attention (tokens don't attend within their own view)
        2. Allows cross-view attention only between tokens within distance_threshold

        Args:
            coords3d: (B, total_tokens, 3)
            tokens_per_view: int
            B: batch size

        Returns:
            attn_mask: (B, total_tokens, total_tokens) bool tensor, True = masked
        """
        total_tokens = coords3d.shape[1]
        device = coords3d.device

        # 1. Create self-view mask: block attention within the same view
        view_ids = torch.arange(self.num_views, device=device).repeat_interleave(tokens_per_view)
        same_view = view_ids.unsqueeze(0) == view_ids.unsqueeze(1)  # (total_tokens, total_tokens)
        same_view = same_view.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)

        if self.distance_threshold <= 0:
            # No distance masking — only block self-view
            return same_view

        # 2. Compute pairwise 3D distances for cross-view tokens
        # To save memory, we compute this in chunks per batch
        # coords3d: (B, T, 3)
        # We use squared distance and compare against threshold^2
        threshold_sq = self.distance_threshold ** 2

        # (B, T, 1, 3) - (B, 1, T, 3) → (B, T, T)
        # This is O(T^2) memory. For T ~ 6*23*23 = 3174, this is manageable (~40MB)
        diff = coords3d.unsqueeze(2) - coords3d.unsqueeze(1)  # (B, T, T, 3)
        dist_sq = (diff ** 2).sum(dim=-1)  # (B, T, T)

        too_far = dist_sq > threshold_sq  # (B, T, T)

        # Combine: mask if same view OR too far
        attn_mask = same_view | too_far

        # Safety: if a token has ALL positions masked, softmax produces NaN.
        # Unmask the diagonal (self-attention fallback) for fully-masked rows.
        all_masked = attn_mask.all(dim=-1)  # (B, T)
        if all_masked.any():
            # Get (batch_idx, token_idx) pairs for fully-masked rows
            b_idx, t_idx = all_masked.nonzero(as_tuple=True)
            # Unmask attn_mask[b, t, t] — allow self-attention as identity fallback
            attn_mask[b_idx, t_idx, t_idx] = False

        return attn_mask


class CrossView3DAttentionLayer(nn.Module):
    """Single layer of cross-view 3D-aware attention with residual + FFN."""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.0, ffn_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (B, T, C)
            attn_mask: (B, T, T) bool, True = masked
        """
        # Pre-norm self-attention with mask
        residual = x
        x_normed = self.norm1(x)

        if attn_mask is not None:
            # nn.MultiheadAttention with batch_first=True expects:
            # attn_mask: (B*num_heads, T, T) or (T, T) — True means "ignore"
            # We need to expand for num_heads
            B, T, _ = x.shape
            num_heads = self.attn.num_heads
            # Expand: (B, T, T) → (B * num_heads, T, T)
            attn_mask_expanded = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
            attn_mask_expanded = attn_mask_expanded.reshape(B * num_heads, T, T)
        else:
            attn_mask_expanded = None

        attn_out, _ = self.attn(
            x_normed, x_normed, x_normed,
            attn_mask=attn_mask_expanded,
        )
        x = residual + attn_out

        # Pre-norm FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x
