import numpy as np
import torch
from sklearn.metrics import pairwise_distances


# ── VP-SDE noise schedule ─────────────────────────────────────────────────────

def marginal_prob_std(t):
    """Standard deviation of the perturbation kernel for the VP SDE."""
    beta_0, beta_1    = 0.1, 20.0
    log_mean_coeff    = -0.25 * t**2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    return torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))


# ── Generation quality metrics ────────────────────────────────────────────────

def compute_mmd(x, y, sigma=None):
    """Gaussian-kernel MMD between two sample sets with optional median-heuristic bandwidth."""
    xx = pairwise_distances(x, x, squared=True)
    yy = pairwise_distances(y, y, squared=True)
    xy = pairwise_distances(x, y, squared=True)
    if sigma is None:
        tri    = xy[np.triu_indices_from(xy, k=1)] if xy.shape[0] == xy.shape[1] else xy.ravel()
        med    = np.median(tri[tri > 0]) if np.any(tri > 0) else 1.0
        sigma2 = max(med, 1e-6)
    else:
        sigma2 = float(sigma)**2
    k_xx = np.exp(-xx / (2 * sigma2)).mean()
    k_yy = np.exp(-yy / (2 * sigma2)).mean()
    k_xy = np.exp(-xy / (2 * sigma2)).mean()
    return k_xx + k_yy - 2 * k_xy


def compute_tvd_hist2d(x, y, bins=80, lim=6.0):
    """Empirical total variation distance between two 2D point clouds via shared histogram bins."""
    edges = [np.linspace(-lim, lim, bins + 1), np.linspace(-lim, lim, bins + 1)]
    hx, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=edges)
    hy, _, _ = np.histogram2d(y[:, 0], y[:, 1], bins=edges)
    px = hx / (hx.sum() + 1e-12)
    py = hy / (hy.sum() + 1e-12)
    return 0.5 * np.abs(px - py).sum()


def hist_density(samples, bins=80, lim=7.2):
    """Normalized 2D histogram density array (row = y, col = x)."""
    edges = np.linspace(-lim, lim, bins + 1)
    h, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[edges, edges])
    return (h / (h.sum() + 1e-12)).T


def estimate_score_error(model, dataset, device, n_eval=2048, weighted=True, return_relative=True):
    """Practical epsilon_theta proxy aligned with the VP denoising score matching objective."""
    model.eval()
    with torch.no_grad():
        x0  = dataset[torch.randperm(dataset.shape[0])[:n_eval]].to(device)
        t   = torch.rand(x0.shape[0], device=device) * 0.9999 + 1e-4
        std = marginal_prob_std(t).unsqueeze(1)
        z   = torch.randn_like(x0)
        xt  = x0 * torch.sqrt(1 - std**2) + z * std
        target_score = -z / std
        pred_score   = model(xt, t)
        residual   = (pred_score - target_score) * std if weighted else (pred_score - target_score)
        target_ref = target_score * std if weighted else target_score
        rmse = torch.sqrt(torch.mean(residual**2)).item()
        if return_relative:
            return rmse / (torch.sqrt(torch.mean(target_ref**2)).item() + 1e-8)
        return rmse


# ── Sampling ──────────────────────────────────────────────────────────────────

def reverse_sde_sampler(model, device, n_samples=1000, n_steps=500, curl_factor=0.0):
    """Euler-Maruyama reverse-SDE sampler with optional non-conservative curl injection."""
    model.eval()
    with torch.no_grad():
        x          = torch.randn(n_samples, 2).to(device)
        time_steps = torch.linspace(1.0, 1e-4, n_steps).to(device)
        dt         = time_steps[0] - time_steps[1]
        beta_0, beta_1 = 0.1, 20.0
        for t in time_steps:
            batch_t = torch.ones(n_samples, device=device) * t
            beta_t  = beta_0 + t * (beta_1 - beta_0)
            score   = model(x, batch_t)
            if curl_factor > 0:
                score = score + curl_factor * torch.stack([-x[:, 1], x[:, 0]], dim=1)
            drift     = -0.5 * beta_t * x - beta_t * score
            diffusion = torch.sqrt(beta_t)
            z = torch.randn_like(x) if t > 1e-4 else torch.zeros_like(x)
            x = x - drift * dt + diffusion * torch.sqrt(dt) * z
    return x.cpu().numpy()


def reverse_ode_sampler_with_grid(model, device, time_grid, n_samples=1200):
    """Deterministic reverse-ODE sampler driven by a user-supplied time grid."""
    x = torch.randn(n_samples, 2, device=device)
    beta_0, beta_1 = 0.1, 20.0
    for i in range(len(time_grid) - 1):
        t_curr  = float(time_grid[i])
        dt      = max(t_curr - float(time_grid[i + 1]), 1e-8)
        beta_t  = beta_0 + t_curr * (beta_1 - beta_0)
        t_batch = torch.full((n_samples,), t_curr, device=device)
        score   = model(x, t_batch)
        x       = x + 0.5 * beta_t * (x + score) * dt
    return x.detach().cpu().numpy()


def evaluate_generation_metrics(model, device, target_np, n_samples=1000, n_steps=200, curl_factor=0.0):
    """Generate samples from the reverse SDE and return (samples, MMD, TVD)."""
    gen = reverse_sde_sampler(model, device, n_samples=n_samples, n_steps=n_steps, curl_factor=curl_factor)
    return gen, compute_mmd(gen, target_np), compute_tvd_hist2d(gen, target_np)


def evaluate_generation_metrics_ode(model, device, target_np, time_grid, n_samples=1200, bins=90, lim=7.2):
    """Generate samples from the reverse ODE and return (samples, MMD, TVD)."""
    gen = reverse_ode_sampler_with_grid(model, device, time_grid=time_grid, n_samples=n_samples)
    return gen, compute_mmd(gen, target_np), compute_tvd_hist2d(gen, target_np, bins=bins, lim=lim)


def sample_forward_noised(data_np, device, t_val, n=3000):
    """Return forward-noised samples x_t ~ q(x_t | x_0) for visualization."""
    x0  = torch.tensor(data_np[:n], dtype=torch.float32, device=device)
    std = marginal_prob_std(
        torch.full((x0.shape[0],), float(t_val), device=device)
    ).unsqueeze(1).cpu().numpy()
    mean_coeff = np.sqrt(1.0 - std**2)
    return x0.cpu().numpy() * mean_coeff + np.random.randn(x0.shape[0], 2) * std


# ── Scheduling ────────────────────────────────────────────────────────────────

def build_time_grid(schedule_name, n_steps, t_min=0.02, kappa=1.0, adiabatic_profile=None):
    """Build a diffusion time grid for a given schedule type."""
    n_steps = int(n_steps)
    u = np.linspace(0.0, 1.0, n_steps + 1)
    if schedule_name == "linear":
        t = np.linspace(1.0, t_min, n_steps + 1)
    elif schedule_name == "cosine":
        t = np.cos(np.linspace(0.0, np.arccos(t_min), n_steps + 1))
    elif schedule_name == "kappa":
        t = t_min + (1.0 - t_min) * ((1.0 - u) ** float(kappa))
    elif schedule_name == "adiabatic":
        if adiabatic_profile is None:
            raise ValueError("adiabatic_profile must be provided for schedule_name='adiabatic'")
        q = np.linspace(0.0, 1.0, n_steps + 1)
        t = np.interp(q, adiabatic_profile["cum_cost"], adiabatic_profile["t_probe_desc"])
    else:
        raise ValueError(f"Unknown schedule_name: {schedule_name!r}")
    t = np.clip(np.asarray(t, dtype=float), t_min, 1.0)
    t[0] = 1.0
    for i in range(1, len(t)):
        if t[i] > t[i - 1]:
            t[i] = t[i - 1]
    t[-1] = t_min
    return t


def estimate_score_velocity_variance_rhot(model, dataset, device, t_val, n_probe_samples=768, dt_fd=2e-3):
    """Finite-difference estimate of Var_{rho_t}[partial_t S_theta] for adiabatic scheduling."""
    n   = min(int(n_probe_samples), int(dataset.shape[0]))
    x0  = dataset[torch.randperm(dataset.shape[0])[:n]].to(device)
    std = marginal_prob_std(torch.full((n,), float(t_val), device=device)).unsqueeze(1)
    xt  = x0 * torch.sqrt(1.0 - std**2) + std * torch.randn_like(x0)
    t_plus   = min(1.0,  float(t_val) + dt_fd)
    t_minus  = max(1e-4, float(t_val) - dt_fd)
    dt_actual = max(t_plus - t_minus, 1e-6)
    with torch.no_grad():
        s_plus  = model(xt, torch.full((n,), t_plus,  device=device))
        s_minus = model(xt, torch.full((n,), t_minus, device=device))
    dsdt = (s_plus - s_minus) / dt_actual
    return float(torch.mean(torch.sum(dsdt**2, dim=1)).item())


def build_adiabatic_profile_4b(model, dataset, device, t_min=0.02, n_probe_t=50, alpha_topo=0.70, gap_floor=1e-4):
    """Build a spectral-cost adiabatic profile for the hierarchical-GMM experiment."""
    from .hamiltonian import estimate_hamiltonian_spectrum
    t_probe_desc = np.linspace(1.0, t_min, int(n_probe_t))
    raw_cost, gaps, vel_vars = [], [], []
    for t in t_probe_desc:
        _, g, _ = estimate_hamiltonian_spectrum(model, device, t_eval=float(t), N=42, lim=7.2, k=5)
        g    = max(float(g), float(gap_floor))
        vvar = estimate_score_velocity_variance_rhot(
            model, dataset, device, float(t), n_probe_samples=896, dt_fd=2e-3
        )
        raw_cost.append(float(np.sqrt(max(vvar, 1e-12)) / (g ** 1.5)))
        gaps.append(float(g))
        vel_vars.append(float(vvar))

    raw_cost = np.asarray(raw_cost, dtype=float)
    smooth   = raw_cost.copy()
    if len(smooth) >= 3:
        smooth[1:-1] = (raw_cost[:-2] + raw_cost[1:-1] + raw_cost[2:]) / 3.0
    smooth = np.minimum(smooth, np.quantile(smooth, 0.90))
    smooth = np.log1p(smooth)

    dt_seg       = np.abs(np.diff(t_probe_desc)) + 1e-12
    topo_norm    = np.sum(smooth[:-1] * dt_seg) + 1e-12
    topo_density = smooth / topo_norm
    kin_density  = np.ones_like(t_probe_desc) / (1.0 - t_min)
    blend        = alpha_topo * topo_density + (1.0 - alpha_topo) * kin_density
    mid          = 0.5 * (blend[:-1] + blend[1:])
    cum          = np.concatenate([[0.0], np.cumsum(mid * dt_seg)])
    cum          = cum / (cum[-1] + 1e-12)

    return {
        't_probe_desc': t_probe_desc,
        'gap':          np.asarray(gaps),
        'vel_var':      np.asarray(vel_vars),
        'raw_cost':     raw_cost,
        'blend_density': blend,
        'cum_cost':     cum,
    }


def estimate_curl_rms(model, device, n_pts=256, t_eval=0.2):
    """RMS curl of the score field — nonzero curl indicates non-conservativity."""
    x   = torch.randn(n_pts, 2, device=device, requires_grad=True)
    t   = torch.full((n_pts,), float(t_eval), device=device)
    s   = model(x, t)
    dsx = torch.autograd.grad(s[:, 0].sum(), x, retain_graph=True)[0]
    dsy = torch.autograd.grad(s[:, 1].sum(), x)[0]
    curl = dsy[:, 0] - dsx[:, 1]
    return float(torch.sqrt(torch.mean(curl**2)).item())


# ── Numerical error utilities ─────────────────────────────────────────────────

def relative_error_percent(estimate, reference):
    return 100.0 * abs(estimate - reference) / abs(reference)


def root_mean_square_error(estimate, reference):
    return float(np.sqrt(np.mean((np.asarray(estimate) - np.asarray(reference))**2)))


def partial_corr(x, y, z):
    """Pearson r(x, y | z): partial correlation after linearly removing z from both x and y."""
    def resid(a, b):
        c = np.polyfit(b, a, 1)
        return a - (c[0] * b + c[1])
    return float(np.corrcoef(resid(x, z), resid(y, z))[0, 1])


def get_energy(spectrum, state_key):
    s = spectrum.get(state_key)
    return np.nan if s is None else s["E"]


def fmt_energy(v):
    return "unbound" if np.isnan(v) else f"{v:+.4f}"


def fmt_error(v):
    return "unbound" if np.isnan(v) else f"{v:.4f}"
