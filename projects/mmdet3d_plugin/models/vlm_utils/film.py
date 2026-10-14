"""FiLM injection of UniDepth features into intermediate LLM decoder layers (vision tokens only).

For every vision token a condition vector is built outside the LLM from frozen UniDepth outputs (no grad):
pooled decoder latents, log min-depth and log confidence of the token's patch. Before each selected decoder
layer l the vision-token hidden states are modulated as h <- h * (1 + gamma_l) + beta_l, with
(gamma_l, beta_l) = MLP_l(LN_l(cond)) and a zero-initialised last layer (identity at start).

Gradient checkpointing safety: the per-layer MLP runs inside a forward-pre-hook of the decoder layer, so it
is recomputed inside the checkpointed segment and only captures leaf parameters plus the no-grad condition.
Every parameter belongs to exactly one layer, so DDP sees one gradient per parameter (no parameter is shared
across checkpointed segments).
"""
from functools import partial

import torch
import torch.nn as nn


class DepthFiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim, layers=(7, 14, 21), mlp_dim=256):
        super().__init__()
        self.layers = [int(l) for l in layers]
        self.nets = nn.ModuleDict()
        for l in self.layers:
            net = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, mlp_dim), nn.SiLU(),
                                nn.Linear(mlp_dim, 2 * hidden_dim))
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.nets[str(l)] = net
        self._state = None
        self._handles = []
        self.hits = 0  # number of applied modulations (sanity check that the hooks fire)

    def attach(self, decoder_layers):
        for l in self.layers:
            self._handles.append(decoder_layers[l].register_forward_pre_hook(partial(self._hook, l)))

    def set_state(self, cond, mask):
        """cond (num_vis_tokens_in_batch, cond_dim), no grad, ordered like hidden[mask];
        mask (B, L) bool of the vision-token positions."""
        self._state = (cond.detach(), mask)

    def clear(self):
        self._state = None

    def _hook(self, layer, module, args):
        if self._state is None or not args:
            return None
        cond, mask = self._state
        h = args[0]
        if h.dim() != 3 or h.shape[:2] != mask.shape:  # incremental decoding steps: nothing to modulate
            return None
        g, b = self.nets[str(layer)](cond.float()).chunk(2, dim=-1)
        self.hits += 1
        sel = h[mask]
        new = sel * (1 + g.to(h.dtype)) + b.to(h.dtype)
        return (h.masked_scatter(mask.unsqueeze(-1), new),) + tuple(args[1:])
