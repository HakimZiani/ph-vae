import torch
import torch.nn.functional as F



def _ph_logpdf_uniformized(alpha, T, t, x, eps=1e-8, K_max=500):
    """
    Log-pdf of a Phase-Type distribution, evaluated via uniformization.

    Uniformization rewrites exp(T*x) as a Poisson-weighted sum of matrix
    powers of the discrete transition matrix P = I + T/lambda, which is
    numerically stable and fully differentiable.

    Args:
        alpha : (B, m)     initial distribution over transient states
        T     : (B, m, m)  sub-generator matrix  (diagonal entries are negative)
        t     : (B, m)     exit-rate vector  t = -T @ 1
        x     : (B,)       observation values  (must be > 0)
        eps   : truncation tolerance for the Poisson tail
        K_max : hard cap on the number of Poisson terms

    Returns:
        log_f : (B,)  log-pdf values
    """
    B, m = alpha.shape

    lambda_ = (-torch.diagonal(T, dim1=-2, dim2=-1)).max(dim=-1).values  # (B,)
    P = torch.eye(m, device=T.device, dtype=T.dtype).unsqueeze(0) + T / lambda_.view(B, 1, 1)  # (B, m, m)
    lam_x = lambda_ * x  # (B,)

    log_wk = -lam_x                        
    cum_mass = torch.exp(log_wk).clone()
   
    log_factorials = torch.zeros(K_max + 1, device=T.device, dtype=T.dtype)
    for k in range(1, K_max + 1):
        log_factorials[k] = log_factorials[k-1] + torch.log(torch.tensor(k, dtype=T.dtype, device=T.device))
   
    K = 0
    for k in range(1, K_max + 1):
        log_wk = log_wk + torch.log(lam_x) - log_factorials[k]
        cum_mass = cum_mass + torch.exp(log_wk)
        if torch.all(cum_mass >= 1.0 - eps):
            K = k
            break
    else:
        K = K_max

   
    log_wk = -lam_x                          # (B,)
    alpha_Pk = alpha.clone()                 # (B, m) 
    S = torch.exp(log_wk).unsqueeze(-1) * alpha_Pk  # (B, m)

    for k in range(1, K + 1):
           
           alpha_Pk = torch.bmm(alpha_Pk.unsqueeze(1), P).squeeze(1)
           log_wk = log_wk + torch.log(lam_x) - log_factorials[k]
           S = S + torch.exp(log_wk).view(B, 1) * alpha_Pk

    f = (S * t).sum(dim=-1)                  # (B,)
    return torch.log(f.clamp_min(1e-12))     # (B,)




def gaussian_kl(mu, logvar):
    """
    KL( N(mu, diag(exp(logvar))) || N(0, I) )

    Args:
        mu     : (B, d)
        logvar : (B, d)

    Returns:
        kl : (B,)  per-sample KL
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)




def _ph_logpdf_one_dim(alpha_j, T_j, t_j, x_j, uniformization=False):
    """
    Log-pdf for dimension j across the batch.

    Args:
        alpha_j : (B, m)
        T_j     : (B, m, m)
        t_j     : (B, m)
        x_j     : (B,)

    Returns:
        log_f_j : (B,)
    """
    if not uniformization:
        return _ph_logpdf_matrix_exp(alpha_j, T_j, t_j, x_j.clamp_min(1e-8))
    else:
        return _ph_logpdf_uniformized(alpha_j, T_j, t_j, x_j.clamp_min(1e-8))


def elbo_phvae_multidim(model, x, beta=1.0):
    """
    ELBO for the multivariate PH-VAE.

    The reconstruction term sums log-pdfs across all D dimensions
    (conditional independence given z, see Eq. 4 in the paper).
    Each dimension uses uniformization for stable likelihood evaluation.

    Args:
        model : PH-VAE model, returns (mu, logvar, alphas, Ts, ts)
                where alphas, Ts, ts are lists of length D,
                each element having shape (B, m), (B, m, m), (B, m)
        x     : (B, D)  observations
        beta  : KL weight

    Returns:
        loss            : scalar  (negative ELBO)
        recon_logprob   : scalar  (mean total reconstruction log-prob, detached)
        kl              : scalar  (mean KL, detached)
    """
    mu, logvar, alphas, Ts, ts = model(x)

    x_pos = x.clamp_min(1e-8)   # (B, D)
    D = x_pos.shape[1]

    
    recon_logprob = sum(
        _ph_logpdf_one_dim(alphas[j], Ts[j], ts[j], x_pos[:, j])
        for j in range(D)
    )  # (B,)

    kl = gaussian_kl(mu, logvar)  # (B,)

    loss = -(recon_logprob - beta * kl).mean()

    return loss, recon_logprob.mean().detach(), kl.mean().detach()




def _ph_logpdf_matrix_exp(alpha, T, t, x):
    B, m = alpha.shape
    Tx = T * x.view(B, 1, 1)
    expTx = torch.linalg.matrix_exp(Tx)
    tmp = torch.bmm(alpha.unsqueeze(1), expTx).squeeze(1)
    f = (tmp * t).sum(dim=-1)
    return torch.log(f.clamp_min(1e-12))