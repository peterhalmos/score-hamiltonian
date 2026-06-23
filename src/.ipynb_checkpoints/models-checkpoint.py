import numpy as np
import torch
import torch.nn as nn

try:
    from nflows.flows.base import Flow
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.coupling import AffineCouplingTransform
    from nflows.transforms.permutations import RandomPermutation
    HAS_NFLOWS = True
except Exception:
    HAS_NFLOWS = False


# ── Shared utility ────────────────────────────────────────────────────────────

class NetWrapper(nn.Module):
    """Adapts nn.Sequential to the nflows coupling-layer interface."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, context=None):
        return self.net(x)


# ── Hydrogen Atom ─────────────────────────────────────────────────────────────

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim=64, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, sigma):
        proj = sigma[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class QuantumStateNet(nn.Module):
    """log rho(x, sigma) = MLP_correction(x, sigma) - 2 Z_eff ||x||
    """
    def __init__(self):
        super().__init__()
        self.embed = GaussianFourierProjection(embed_dim=64, scale=30.0)
        self.net = nn.Sequential(
            nn.Linear(3 + 64, 256), nn.SiLU(),
            nn.Linear(256, 256),    nn.SiLU(),
            nn.Linear(256, 256),    nn.SiLU(),
            nn.Linear(256, 1),
        )
        self.log_Z = nn.Parameter(torch.tensor(0.1)) # Z_eff away from target at init; must be learned
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x, sigma):
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(1)
        r = (x**2).sum(dim=1, keepdim=True).sqrt() + 1e-9
        h = torch.cat([x, self.embed(sigma.squeeze())], dim=1)
        return self.net(h) - 2.0 * torch.exp(self.log_Z) * r
    
    def get_physics(self, x, sigma_val=0.0):
        """Returns (score S, quantum potential Q) at the given noise level."""
        x = x.requires_grad_(True)
        n = x.shape[0]
        sig = torch.full(
            (n, 1),
            float(sigma_val) if not isinstance(sigma_val, torch.Tensor) else sigma_val.item(),
            device=x.device,
        )
        log_rho = self.forward(x, sig)
        score = torch.autograd.grad(log_rho.sum(), x, create_graph=True)[0]
        div_s = sum(
            torch.autograd.grad(score[:, i].sum(), x, create_graph=True)[0][:, i]
            for i in range(3)
        )
        s_sq = (score**2).sum(dim=1)
        Q = -0.25 * div_s - 0.125 * s_sq
        return score, Q


# ── Coupled Harmonic Oscillator ───────────────────────────────────────────────

class SteadyStateScoreNet(nn.Module):
    """MLP score network approximating grad log rho(x) for 2D distributions.
    No time conditioning; uses Softplus activations.
    """
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


# ── SpectralTest diffusion models ─────────────────────────────────────────────

class TimedScoreNet(nn.Module):
    """Time-conditioned MLP score network: input is (x, t) concatenated -> R^3."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if t.dim() == 0:
            t = t.repeat(x.shape[0]).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        elif t.dim() == 2 and t.shape[1] == 1:
            pass
        else:
            t = t.reshape(-1, 1)
        if t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0], 1)
        return self.net(torch.cat([x, t], dim=1))


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal Fourier features for scalar time t -> R^(2*n_freq)."""
    def __init__(self, n_freq=16):
        super().__init__()
        freqs = torch.exp(torch.linspace(0.0, np.log(100.0), n_freq))
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        t = t.unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


class ConservativeScoreNet(nn.Module):
    """Conservative score network: score(x,t) = -grad_x Phi_theta(x,t).
    Uses Fourier time embedding, LayerNorm, and is trained with EMA weights.
    """
    def __init__(self, hidden_dim=192, n_freq=16):
        super().__init__()
        self.time_emb = FourierTimeEmbedding(n_freq=n_freq)
        in_dim = 2 + 2 * n_freq
        layers = []
        prev = in_dim
        for _ in range(4):
            layers += [nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def _phi(self, x, t):
        return self.net(torch.cat([x, self.time_emb(t)], dim=-1))

    def forward(self, x, t):
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            if t.dim() == 0:
                t = t.repeat(x.shape[0])
            elif t.dim() == 2:
                t = t.squeeze(1)
            if t.shape[0] != x.shape[0]:
                t = t.expand(x.shape[0])
            phi = self._phi(x, t)
            score = -torch.autograd.grad(
                phi.sum(), x, create_graph=True, retain_graph=True
            )[0]
        return score


# ── Eigenmodes ────────────────────────────────────────────────────────────────

class ScoreNet2D(nn.Module):
    """Simple 2D score MLP without time conditioning. Uses SiLU activations."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)
