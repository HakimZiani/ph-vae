
import torch
import torch.nn.functional as F
# sampling function for PH
def ph_sample_batch(alpha, T, t):
        device = alpha.device
        B, m = alpha.shape

        u = torch.rand(B, device=device)
        cdf = torch.cumsum(alpha, dim=1)  # [B,m]
        init_state = torch.sum(u.unsqueeze(1) > cdf, dim=1)  # [B], int64

        r_all = -(T.diagonal(dim1=-2, dim2=-1))  # [B, m]
        off = T.clone()
        off.diagonal(dim1=-2, dim2=-1).zero_()   # [B, m, m]

        # simulation
        times = torch.zeros(B, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)
        state = init_state.clone()  # [B]

        max_steps = 10000
        step = 0

        while active.any() and step < max_steps:
            step += 1
            a_idx = torch.where(active)[0]           
            s = state[a_idx]                 
            r = r_all[a_idx, s]                  

            u1 = torch.rand_like(r).clamp_min(1e-12)
            dt = -torch.log(u1) / r
            times[a_idx] += dt

            row_off = off[a_idx, s, :]               
            exit_r = t[a_idx, :]                    
            
            prob_off = row_off / r.unsqueeze(1)      
            prob_abs = exit_r.gather(1, s.unsqueeze(1)).squeeze(1) / r 

            
            probs = torch.cat([prob_off, prob_abs.unsqueeze(1)], dim=1)  # [Ba, m+1]
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
            cdf2 = torch.cumsum(probs, dim=1)

            u2 = torch.rand(len(a_idx), device=device).unsqueeze(1)
            choice = torch.sum(u2 > cdf2, dim=1)     # in {0..m} ; m == absorb

            # next states where not absorbed
            not_abs = choice < m
            idx_not_abs = a_idx[not_abs]
            next_state = choice[not_abs]             # [Bna]
            state[idx_not_abs] = next_state

            # absorbed become inactive
            idx_abs = a_idx[~not_abs]
            active[idx_abs] = False

        return times
@torch.no_grad()
def ph_sample_multidim(alphas, As, ts):
    """
    Sample from k independent PH distributions.

    Inputs:
        alphas: list of length k, each [B, m]
        As:     list of length k, each [B, m, m]
        ts:     list of length k, each [B, m]

    Returns:
        samples: [B, k]
    """
    samples = []

    for alpha, A, t in zip(alphas, As, ts):
        xj = ph_sample_batch(alpha, A, t)  # [B]
        samples.append(xj.unsqueeze(1))

    return torch.cat(samples, dim=1)  # [B, k]


def gumbel_softmax_sample(logits, tau=0.5, hard=False):
    """
    Differentiably samples from a categorical distribution using the Gumbel-Softmax trick.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y_soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)
    
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
        
    return y
def sample_phs(alpha, A, t):
    """
    alpha: [B, n, m]
    A:     [B, n, m, m]
    t:     [B, n, m]
    returns samples: [B, n]
    """
    B, n, m, _ = A.shape
    eps = 1e-8
    
    
    
    lambdas = -torch.diagonal(A, dim1=2, dim2=3)
    
    U = torch.rand(B, n, m, device=alpha.device)
    U = U.clamp(min=eps, max=1 - eps)
    # apply reparam trick
    Z = -torch.log(U) / lambdas 


    alpha_onehot = gumbel_softmax_sample(alpha, tau=0.5, hard=True)
    mask_alpha = torch.cummax(alpha_onehot[0], dim=-1).values
    Z = Z * mask_alpha

    samples = Z.sum(dim=-1) 
    return samples         
    