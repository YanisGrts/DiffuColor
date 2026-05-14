import torch
import math


def cosine_schedule(T):
    steps = torch.arange(T + 1) / T
    f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
    alpha_bars = f / f[0]
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    return torch.clamp(betas, 0, 0.999), alpha_bars[1:]


def q_sample(ab, t, noise, alpha_bars):
    """Add noise to ab at timestep t (forward process)."""
    ab_t = alpha_bars[t].sqrt().view(-1, 1, 1, 1)
    noise_t = (1 - alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
    return ab_t * ab + noise_t * noise


def p_sample(model, ab_noisy, L, t, alpha_bars, betas, alphas,  guidance_scale=1.0):
    """One denoising step (reverse process)."""
    with torch.no_grad():
        pred_noise_cond = model(ab_noisy, L, t)
        if guidance_scale != 1.0:
            # Unconditional prediction (L zeroed out)
            L_null = torch.zeros_like(L)
            pred_noise_uncond = model(ab_noisy, L_null, t)

            # CFG blend
            pred_noise = pred_noise_uncond + guidance_scale * \
                (pred_noise_cond - pred_noise_uncond)
        else:
            pred_noise = pred_noise_cond

        device = ab_noisy.device
        b = betas[t].view(-1, 1, 1, 1)
        a = alphas[t].view(-1, 1, 1, 1)
        ab_bar = alpha_bars[t].view(-1, 1, 1, 1)

        # Reconstruct x0 from predicted noise
        pred_x0 = torch.clamp((ab_noisy - (1 - ab_bar).sqrt() * pred_noise) /
                               ab_bar.sqrt(), -1.0, 1.0)
        # Posterior mean using x0
        ab_bar_prev = torch.where(t > 0, alpha_bars[t - 1], torch.ones_like(alpha_bars[t])).view(-1, 1, 1, 1)

        mean = (b * ab_bar_prev.sqrt() / (1 - ab_bar)) * pred_x0 + \
               ((1 - ab_bar_prev) * a.sqrt() / (1 - ab_bar)) * ab_noisy
        noise = torch.randn_like(ab_noisy) if t[0] > 0 else 0
        return mean + b.sqrt() * noise


T = 1000
# betas  = torch.linspace(1e-4, 0.02, T)          # noise schedule
# alpha_bars = torch.cumprod(alphas, dim=0)        # ᾱ_t
betas, alpha_bars = cosine_schedule(T=1000)
alphas = 1.0 - betas
