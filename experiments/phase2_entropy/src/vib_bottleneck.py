"""VIB bottleneck module — inserted at a single transformer layer during inference.

Based on AdaVIB (Yocam et al., 2025) architecture, adapted for pure text LLM:
- mu_proj / logvar_proj: d_model -> d_bottleneck
- out_proj: d_bottleneck -> d_model (residual delta)
- Training: CE + beta * KL(N(mu,sigma) || N(0,I))
- Inference: deterministic (mu only, no noise)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VIBBottleneck(nn.Module):
    def __init__(self, d_model: int = 2048, d_bottleneck: int = 512):
        super().__init__()
        self.d_model = d_model
        self.d_bottleneck = d_bottleneck

        self.mu_proj = nn.Linear(d_model, d_bottleneck)
        self.logvar_proj = nn.Linear(d_model, d_bottleneck)
        self.out_proj = nn.Linear(d_bottleneck, d_model)

        self._init_weights()
        self._last_kl = torch.tensor(0.0)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.mu_proj.weight)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.xavier_uniform_(self.logvar_proj.weight)
        nn.init.zeros_(self.logvar_proj.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @property
    def last_kl(self) -> torch.Tensor:
        return self._last_kl

    def forward(self, h: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        """Apply VIB bottleneck to hidden states.

        Args:
            h: [batch, seq, d_model] hidden states at the hook point.
            is_training: if True, use reparameterization (mu + eps*std).
                         if False, use deterministic mu.

        Returns:
            modified_h: [batch, seq, d_model] = h + out_proj(z).
        """
        # Cast to float32 for numerical stability (fp16 exp/pow overflow easily)
        h_dtype = h.dtype
        h_f32 = h.to(torch.float32)

        mu = self.mu_proj(h_f32)
        logvar = self.logvar_proj(h_f32)
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)

        if is_training:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        delta = self.out_proj(z)

        # KL(N(mu, sigma^2) || N(0, I)) per dimension, averaged over batch & seq
        # Stable: 0.5 * (sigma^2 + mu^2 - 1 - log(sigma^2))
        sigma2 = logvar.exp()
        kl_per_dim = 0.5 * (sigma2 + mu.pow(2) - 1 - logvar)
        self._last_kl = kl_per_dim.sum(dim=-1).mean()

        return h + delta.to(h_dtype)

    def make_hook(self, is_training: bool = True):
        """Return a TransformerLens-compatible hook function.

        The hook intercepts the residual stream and applies the VIB bottleneck.
        """

        def hook(activation, hook=None):
            return self.forward(activation, is_training=is_training)

        return hook
