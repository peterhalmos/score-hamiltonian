import numpy as np
import torch
import torch.nn.functional as F


def sample_hydrogen_ground_state(n_samples):
    """Exact samples from |psi_1s|^2 proportional to r^2 exp(-2r) via the Gamma trick."""
    r     = np.random.gamma(shape=3.0, scale=0.5, size=n_samples)
    phi   = 2 * np.pi * np.random.uniform(0, 1, n_samples)
    theta = np.arccos(2 * np.random.uniform(0, 1, n_samples) - 1)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32)


def sample_2d_gmm(n_samples, mode_distance=2.0):
    """Samples from a symmetric 2D Gaussian mixture with two modes."""
    centers = np.array([[-mode_distance, 0.0], [mode_distance, 0.0]])
    choices = np.random.randint(0, 2, size=n_samples)
    noise   = np.random.randn(n_samples, 2) * 0.5
    return torch.tensor(centers[choices] + noise, dtype=torch.float32)


def sample_hierarchical_gmm(
    n_samples,
    macro_centers=None,
    micro_per_macro=3,
    macro_std=0.0,
    micro_radius=0.9,
    micro_std=0.16,
    macro_weights=None,
    jitter_angle=True,
):
    """Two-level 2D mixture: sparse macro-centers, each with a local ring of micro-modes."""
    if macro_centers is None:
        macro_centers = np.array([[-3.0, -3.0], [3.0, -3.0], [0.0, 3.0]], dtype=float)
    macro_centers = np.asarray(macro_centers, dtype=float)
    n_macro = macro_centers.shape[0]
    if macro_weights is None:
        macro_weights = np.ones(n_macro, dtype=float) / n_macro
    else:
        macro_weights = np.asarray(macro_weights, dtype=float)
        macro_weights = macro_weights / macro_weights.sum()

    micro_centers, micro_weights = [], []
    for m in range(n_macro):
        phi0 = np.random.uniform(0, 2 * np.pi) if jitter_angle else 0.0
        for k in range(int(micro_per_macro)):
            ang = phi0 + 2 * np.pi * k / float(micro_per_macro)
            c = (macro_centers[m]
                 + micro_radius * np.array([np.cos(ang), np.sin(ang)])
                 + np.random.randn(2) * macro_std)
            micro_centers.append(c)
            micro_weights.append(macro_weights[m] / float(micro_per_macro))

    micro_centers = np.asarray(micro_centers, dtype=float)
    micro_weights = np.asarray(micro_weights, dtype=float)
    micro_weights = micro_weights / micro_weights.sum()
    idx     = np.random.choice(len(micro_centers), size=n_samples, p=micro_weights)
    samples = micro_centers[idx] + np.random.randn(n_samples, 2) * micro_std
    return torch.tensor(samples, dtype=torch.float32)


def get_exact_score(grid_pts, X_pts, sigma):
    """Analytical score of the empirical Gaussian KDE at grid points."""
    grid_t   = torch.tensor(grid_pts, dtype=torch.float32)
    X_t      = torch.tensor(X_pts,    dtype=torch.float32)
    dists_sq = torch.cdist(grid_t, X_t)**2
    weights  = F.softmax(-dists_sq / (2 * sigma**2), dim=1)
    return ((torch.mm(weights, X_t) - grid_t) / (sigma**2)).numpy()


class CoupledOscillators:
    """2D coupled quantum harmonic oscillator. Draws exact Gibbs samples and evaluates the true potential."""
    def __init__(self, k=1.0, coupling=5.0):
        self.k    = k
        self.coup = coupling
        self.omega_1 = np.sqrt(k)
        self.omega_2 = np.sqrt(k + 2 * coupling)
        print(f"Physics initialized | coupling = {coupling}")
        print(f"  symmetric mode frequency : {self.omega_1:.4f}")
        print(f"  relative mode frequency  : {self.omega_2:.4f}")

    def sample(self, n_samples):
        sig1 = 1.0 / np.sqrt(2 * self.omega_1)
        sig2 = 1.0 / np.sqrt(2 * self.omega_2)
        q1   = np.random.normal(0, sig1, n_samples)
        q2   = np.random.normal(0, sig2, n_samples)
        x1   = (q1 + q2) / np.sqrt(2)
        x2   = (q1 - q2) / np.sqrt(2)
        return torch.tensor(np.stack([x1, x2], axis=1), dtype=torch.float32)

    def potential(self, X, Y):
        return 0.5 * self.k * (X**2 + Y**2) + 0.5 * self.coup * (X - Y)**2
