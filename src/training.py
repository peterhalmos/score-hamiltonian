import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    from nflows.flows.base import Flow
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.coupling import AffineCouplingTransform
    from nflows.transforms.permutations import RandomPermutation
    HAS_NFLOWS = True
except Exception:
    HAS_NFLOWS = False

from .models import QuantumStateNet, TimedScoreNet, ConservativeScoreNet, NetWrapper
from .utils import marginal_prob_std


def train_dqm(samples, device, epochs=3000, batch=2048):
    """Train QuantumStateNet on hydrogen samples via VP denoising score matching."""
    model = QuantumStateNet().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    data  = samples.to(device)
    N     = len(data)
    sigma_min, sigma_max = 0.01, 0.3

    for step in tqdm(range(epochs), desc="Training score model"):
        idx = torch.randint(0, N, (batch,))
        x0  = data[idx]
        u   = torch.rand(batch, device=device)
        sig = (sigma_min * (sigma_max / sigma_min) ** u).unsqueeze(1)
        xt  = x0 + torch.randn_like(x0) * sig
        xt.requires_grad_(True)
        log_rho  = model(xt, sig)
        pred_s   = torch.autograd.grad(log_rho.sum(), xt, create_graph=True)[0]
        target_s = -(xt - x0) / sig**2
        loss = ((pred_s - target_s)**2 * sig**2).sum(dim=1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0:
            tqdm.write(
                f"  step {step:4d}  loss {loss.item():.4f}  "
                f"Z_eff {torch.exp(model.log_Z).item():.3f}"
            )
    return model


def build_nf_flow(device, dim=3, n_layers=8, hidden=256):
    """Build an affine normalizing flow (nflows) for use as a Boltzmann generator."""
    if not HAS_NFLOWS:
        return None
    base_dist  = StandardNormal(shape=[dim])
    transforms = []
    masks = [torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.0, 1.0, 0.0])]
    for i in range(n_layers):
        mask = masks[i % 2]
        transforms.append(RandomPermutation(features=dim))
        transforms.append(
            AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=lambda in_f, out_f, h=hidden: NetWrapper(
                    nn.Sequential(
                        nn.Linear(in_f, h), nn.SiLU(),
                        nn.Linear(h, h),    nn.SiLU(),
                        nn.Linear(h, h),    nn.SiLU(),
                        nn.Linear(h, out_f),
                    )
                ),
            )
        )
    return Flow(CompositeTransform(transforms), base_dist).to(device)


def train_nf(samples, device, epochs=1500, batch=1024, lambda_cusp=0.05):
    """Train normalizing flow with MLE + Coulomb-cusp score regularization."""
    if not HAS_NFLOWS:
        print("nflows not installed; skipping flow baseline.")
        return None
    flow  = build_nf_flow(device=device, dim=3, n_layers=8, hidden=256)
    opt   = optim.Adam(flow.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    data  = samples.to(device)
    N     = len(data)

    for step in tqdm(range(epochs), desc="Training normalizing flow"):
        idx   = torch.randint(0, N, (batch,))
        x     = data[idx].detach().requires_grad_(True)
        logp  = flow.log_prob(x)
        nll   = -logp.mean()
        score = torch.autograd.grad(logp.sum(), x, create_graph=True)[0]
        r     = (x**2).sum(dim=1, keepdim=True).sqrt() + 1e-8
        target = -2.0 * x / r
        weight = torch.exp(-r.squeeze() / 2.0)
        cusp_loss = (((score - target)**2).sum(dim=1) * weight).mean()
        loss = nll + lambda_cusp * cusp_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 300 == 0:
            tqdm.write(f"  step {step:4d}  NLL {nll.item():.4f}  cusp {cusp_loss.item():.4f}")
    return flow


def train_score_model(model, data, device, epochs=3000):
    """Train a steady-state 2D score network via Hyvärinen implicit score matching."""
    opt      = optim.Adam(model.parameters(), lr=1e-3)
    data_gpu = data.to(device)
    pbar     = tqdm(range(epochs), desc="Training score model")
    for _ in pbar:
        idx    = torch.randint(0, len(data), (1024,))
        x      = data_gpu[idx].detach().requires_grad_(True)
        scores = model(x)
        grad_s1 = torch.autograd.grad(scores[:, 0].sum(), x, create_graph=True)[0][:, 0]
        grad_s2 = torch.autograd.grad(scores[:, 1].sum(), x, create_graph=True)[0][:, 1]
        div_score = grad_s1 + grad_s2
        loss = (div_score + 0.5 * torch.sum(scores**2, dim=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        pbar.set_description(f"Score loss: {loss.item():.4f}")


def train_bg(model, data, device, epochs=3000):
    """Train a Boltzmann generator (normalizing flow) by maximum likelihood."""
    opt      = optim.Adam(model.parameters(), lr=1e-3)
    data_gpu = data.to(device)
    pbar     = tqdm(range(epochs), desc="Training Boltzmann generator")
    for _ in pbar:
        idx  = torch.randint(0, len(data), (1024,))
        x    = data_gpu[idx]
        loss = -model.log_prob(x).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        pbar.set_description(f"BG loss: {loss.item():.4f}")


def build_bg_flow(device, dim=2):
    """Build a 4-layer affine coupling flow for 2D Boltzmann generation."""
    if not HAS_NFLOWS:
        return None
    base_dist  = StandardNormal(shape=[dim])
    transforms = []
    for _ in range(4):
        transforms.append(RandomPermutation(features=dim))
        transforms.append(
            AffineCouplingTransform(
                mask=torch.tensor([1, 0]),
                transform_net_create_fn=lambda in_f, out_f: NetWrapper(
                    nn.Sequential(
                        nn.Linear(in_f, 128), nn.ReLU(),
                        nn.Linear(128, 128),  nn.ReLU(),
                        nn.Linear(128, out_f),
                    )
                ),
            )
        )
    return Flow(CompositeTransform(transforms), base_dist).to(device)


def train_score_network(dataset, device, n_epochs=1500, batch_size=256):
    """Train TimedScoreNet on a 2D dataset via VP denoising score matching."""
    model     = TimedScoreNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset   = dataset.to(device)
    n_samples = dataset.shape[0]

    for epoch in range(n_epochs):
        indices = torch.randperm(n_samples)[:batch_size]
        x_0     = dataset[indices]
        t       = torch.rand(batch_size, device=device) * 0.9999 + 1e-4
        std     = marginal_prob_std(t).unsqueeze(1)
        z       = torch.randn_like(x_0)
        x_t     = x_0 * torch.sqrt(1 - std**2) + z * std
        target_score    = -z / std
        predicted_score = model(x_t, t)
        loss = torch.mean((predicted_score - target_score)**2 * std**2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    return model


def train_conservative_score_network(dataset, device, n_epochs=2000, batch_size=512, lr=8e-4, ema_decay=0.999):
    """Train ConservativeScoreNet with cosine LR and EMA weights. Returns EMA model."""
    model     = ConservativeScoreNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.02)
    ema_params = {k: v.clone().detach() for k, v in model.state_dict().items()}
    data      = dataset.to(device)
    n         = data.shape[0]
    best_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(n, device=device)[:batch_size]
        x0  = data[idx]
        t   = torch.rand(batch_size, device=device) * 0.9999 + 1e-4
        std = marginal_prob_std(t).unsqueeze(1)
        z   = torch.randn_like(x0)
        xt  = x0 * torch.sqrt(1 - std**2) + z * std
        target_score = -z / std
        pred_score   = model(xt, t)
        loss = torch.mean((pred_score - target_score)**2 * std**2)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_params[k] = ema_decay * ema_params[k] + (1.0 - ema_decay) * v.float()
        if loss.item() < best_loss:
            best_loss = loss.item()
        if epoch % 400 == 0 or epoch == n_epochs - 1:
            print(
                f"  epoch {epoch:4d} | loss={loss.item():.4f} | "
                f"best={best_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
            )

    model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in ema_params.items()})
    model.eval()
    print(f"  Training complete. EMA weights loaded. Best loss={best_loss:.4f}")
    return model
