import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.ndimage import gaussian_filter1d


def build_score_hamiltonian_2d(score_field_x, score_field_y, dx, N, wall=500):
    """Assemble H = -0.5 Delta + V_score on an N×N grid with hard-wall BCs."""
    div_S      = np.gradient(score_field_x, dx, axis=1) + np.gradient(score_field_y, dx, axis=0)
    norm_S_sq  = score_field_x**2 + score_field_y**2
    V = 0.25 * div_S + 0.125 * norm_S_sq

    V[:2, :] = wall;  V[-2:, :] = wall
    V[:, :2] = wall;  V[:, -2:] = wall

    ex = np.ones(N)
    D2 = sp.spdiags([ex, -2 * ex, ex], [-1, 0, 1], N, N) / (dx**2)
    Laplacian = sp.kronsum(D2, D2)
    return -0.5 * Laplacian + sp.diags(V.ravel()), V


def estimate_hamiltonian_spectrum(model, device, t_eval=1e-3, N=56, lim=6.0, k=6, curl_factor=0.0):
    """Estimate E0, spectral gap, and low-lying eigenvalues of the Score Hamiltonian."""
    model.eval()
    x_lin = np.linspace(-lim, lim, N)
    dx    = x_lin[1] - x_lin[0]
    xx, yy = np.meshgrid(x_lin, x_lin)
    grid = np.c_[xx.ravel(), yy.ravel()]

    with torch.no_grad():
        xg = torch.tensor(grid, dtype=torch.float32, device=device)
        tg = torch.ones(xg.shape[0], device=device) * float(t_eval)
        s  = model(xg, tg).cpu().numpy()

    sx = s[:, 0].reshape(N, N)
    sy = s[:, 1].reshape(N, N)
    if curl_factor > 0:
        sx = sx + curl_factor * (-yy)
        sy = sy + curl_factor * (xx)

    H, _ = build_score_hamiltonian_2d(sx, sy, dx, N)
    vals, _ = eigsh(H, k=k, which='SA')
    vals = np.sort(vals)
    return float(vals[0]), float(vals[1] - vals[0]), vals


def bohm_potential_grid(model, device, t_val, N=60, lim=7.2):
    """Evaluate the Score/Bohm potential on an N×N grid at diffusion time t_val."""
    x_lin = np.linspace(-lim, lim, N)
    dx    = x_lin[1] - x_lin[0]
    xx, yy = np.meshgrid(x_lin, x_lin)
    grid = np.c_[xx.ravel(), yy.ravel()]
    model.eval()
    with torch.no_grad():
        xg = torch.tensor(grid, dtype=torch.float32, device=device)
        tg = torch.full((xg.shape[0],), float(t_val), device=device)
        s  = model(xg, tg).cpu().numpy()
    sx = s[:, 0].reshape(N, N)
    sy = s[:, 1].reshape(N, N)
    div_s = np.gradient(sx, dx, axis=1) + np.gradient(sy, dx, axis=0)
    return 0.25 * div_s + 0.125 * (sx**2 + sy**2)


def extract_score_hamiltonian_potential_2d(model, device, X, Y):
    """Score Hamiltonian potential on a 2D meshgrid via autograd (exact divergence)."""
    shape  = X.shape
    grid_t = (
        torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
        .to(device)
        .requires_grad_(True)
    )
    scores = model(grid_t)
    grad_s1 = torch.autograd.grad(scores[:, 0].sum(), grid_t, create_graph=True)[0][:, 0]
    grad_s2 = torch.autograd.grad(scores[:, 1].sum(), grid_t, create_graph=True)[0][:, 1]
    div_s   = grad_s1 + grad_s2
    s_sq    = torch.sum(scores**2, dim=1)
    potential = 0.25 * div_s + 0.125 * s_sq
    return potential.detach().cpu().numpy().reshape(shape)


def extract_score_hamiltonian_potential_radial(model, device, r):
    """Score Hamiltonian potential V(r) along the +z axis for a 3D QuantumStateNet."""
    model.eval()
    x_probe = torch.zeros(len(r), 3, device=device)
    x_probe[:, 2] = torch.tensor(r, dtype=torch.float32, device=device)
    with torch.enable_grad():
        _, Q = model.get_physics(x_probe, sigma_val=0.001)
    return -Q.detach().cpu().numpy().flatten()


def extract_thermodynamic_potential(model, device, r):
    """V_thermo(r) = -log rho(r) along the +z axis for a 3D QuantumStateNet."""
    model.eval()
    x_probe = torch.zeros(len(r), 3, device=device)
    x_probe[:, 2] = torch.tensor(r, dtype=torch.float32, device=device)
    sig = torch.full((len(r), 1), 0.001, device=device)
    with torch.no_grad():
        log_rho = model(x_probe, sig)
    return -log_rho.detach().cpu().numpy().flatten()


def extract_flow_potential(flow, device, r):
    """V(r) = -log p_flow(r) along the +z axis for a normalizing flow."""
    if flow is None:
        return None
    flow.eval()
    x_probe = torch.zeros(len(r), 3, device=device)
    x_probe[:, 2] = torch.tensor(r, dtype=torch.float32, device=device)
    with torch.no_grad():
        return -flow.log_prob(x_probe).detach().cpu().numpy().flatten()


def extract_diffusion_thermo_centered(model, device, X, Y):
    """Reconstruct V_thermo = -integral S dx by 2D path integration for a steady-state 2D score model."""
    from scipy.integrate import cumulative_trapezoid
    res       = X.shape[0]
    potential = np.zeros_like(X)
    origin    = torch.zeros(1, 2).to(device)
    bias      = model(origin).detach()

    def get_score(coords):
        return (model(coords) - bias).detach().cpu().numpy()

    x_coords   = X[0, :]
    dx         = x_coords[1] - x_coords[0]
    zeros      = np.zeros_like(x_coords)
    line_x     = torch.tensor(np.stack([x_coords, zeros], axis=1), dtype=torch.float32).to(device)
    s_x        = get_score(line_x)[:, 0]
    center_idx = np.abs(x_coords).argmin()

    phi_x = np.zeros_like(x_coords)
    phi_x[center_idx:] = -cumulative_trapezoid(s_x[center_idx:], dx=dx, initial=0)
    phi_x[:center_idx + 1] = cumulative_trapezoid(
        s_x[:center_idx + 1][::-1], dx=dx, initial=0
    )[::-1]

    y_coords = Y[:, 0]
    for idx, x_coord in enumerate(x_coords):
        line_y = torch.tensor(
            np.stack([np.full(res, x_coord), y_coords], axis=1), dtype=torch.float32
        ).to(device)
        s_y   = get_score(line_y)[:, 1]
        phi_y = np.zeros_like(y_coords)
        phi_y[center_idx:] = -cumulative_trapezoid(s_y[center_idx:], dx=dx, initial=0)
        phi_y[:center_idx + 1] = cumulative_trapezoid(
            s_y[:center_idx + 1][::-1], dx=dx, initial=0
        )[::-1]
        potential[:, idx] = phi_x[idx] + phi_y
    return potential


def get_freqs(model, device, model_type):
    """Recover normal-mode frequencies from the Hessian of the Score potential at the origin."""
    origin = torch.zeros(1, 2).to(device).requires_grad_(True)

    if model_type == "score":
        def score_potential(x):
            scores  = model(x)
            grad_s1 = torch.autograd.grad(scores[:, 0].sum(), x, create_graph=True)[0][:, 0]
            grad_s2 = torch.autograd.grad(scores[:, 1].sum(), x, create_graph=True)[0][:, 1]
            div_s   = grad_s1 + grad_s2
            s_sq    = torch.sum(scores**2, dim=1)
            return (0.25 * div_s + 0.125 * s_sq).sum()
        hessian = torch.autograd.functional.hessian(score_potential, origin).squeeze()
    elif model_type == "diff_thermo":
        jacobian = torch.autograd.functional.jacobian(lambda x: model(x), origin).squeeze()
        hessian  = -jacobian
    elif model_type == "bg":
        hessian = torch.autograd.functional.hessian(
            lambda x: -model.log_prob(x).sum(), origin
        ).squeeze()
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    evals, _ = torch.linalg.eigh(hessian)
    return torch.sqrt(torch.abs(evals)).detach().cpu().numpy()


def solve_schrodinger_1d(r, V_eff):
    """Tridiagonal eigensolver on a 1D radial grid. Returns bound-state energies and wavefunctions."""
    from scipy.linalg import eigh_tridiagonal
    dr   = r[1] - r[0]
    diag = 1.0 / dr**2 + V_eff
    off  = -0.5 / dr**2 * np.ones(len(r) - 1)
    w, v = eigh_tridiagonal(diag, off)
    bound = np.where(w < -0.001)[0]
    if len(bound) == 0:
        return np.array([]), np.zeros((len(r), 0))
    w_b, v_b = w[bound], v[:, bound]
    v_b /= np.sqrt(np.sum(v_b**2 * dr, axis=0))
    return w_b, v_b


def spectrum_from_potential(r, V_raw, method_name, smooth_sigma=0.01):
    """Align tail, smooth, gauge-root to exact 1s ground state, and extract hydrogen spectrum."""
    V = V_raw.copy()
    V -= V[int(0.9 * len(V)):].mean()
    V  = gaussian_filter1d(V, sigma=smooth_sigma)
    energies_1s, _ = solve_schrodinger_1d(r, V)
    if len(energies_1s) == 0:
        raise RuntimeError(f"{method_name}: no bound 1s state found after smoothing/alignment.")
    E_1s_raw    = energies_1s[0]
    gauge_shift = -0.5 - E_1s_raw
    V_rooted    = V + gauge_shift
    print(f"{method_name:<24} raw 1s: {E_1s_raw:+.4f} | gauge shift: {gauge_shift:+.4f}")
    spectrum = {}
    for l in [0, 1, 2]:
        V_eff     = V_rooted + l * (l + 1) / (2 * r**2)
        energies, wfns = solve_schrodinger_1d(r, V_eff)
        for i, E in enumerate(energies):
            if E >= -0.001:
                continue
            n = l + 1 + i
            if n > 4:
                continue
            spectrum[(n, l)] = {"E": E, "E_exact": -0.5 / n**2, "u": wfns[:, i], "r": r}
    return V_rooted, spectrum
