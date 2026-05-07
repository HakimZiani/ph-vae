import time

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .ph_vae_losses import elbo_phvae_multidim



def train_phvae(
    model,
    train_loader,
    epochs=100,
    learning_rate=1e-3,
    beta=1.0,
    grad_clip=None,
    device=None,
    reduce_lr=True,
    lr_patience=5,
    lr_factor=0.1,
    min_delta=0.0,
):
    
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    total_start_time = time.time()
    epoch_times = []
    nll_history = []
    best_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()

        running_loss = 0.0
        running_rec = 0.0
        running_kl = 0.0
        n_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            

            loss, recon_logprob_mean, kl_mean = elbo_phvae_multidim(model, x, beta=beta)

            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            b = x.size(0)
            running_loss += loss.item() * b
            running_rec += recon_logprob_mean.item() * b
            running_kl += kl_mean.item() * b
            n_samples += b

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        epoch_loss = running_loss / max(1, n_samples)
        epoch_rec = running_rec / max(1, n_samples)
        epoch_kl = running_kl / max(1, n_samples)
        nll_history.append(-epoch_rec)

        if reduce_lr:
            if epoch_loss < (best_loss - min_delta):
                best_loss = epoch_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= lr_patience:
                    for g in optimizer.param_groups:
                        g["lr"] *= lr_factor
                    print(
                        "Loss did not improve for "
                        f"{lr_patience} epochs. "
                        f"Reducing learning rate to {optimizer.param_groups[0]['lr']:.2e}"
                    )
                    epochs_since_improvement = 0

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Recon: {epoch_rec:.4f} | "
            f"KL: {epoch_kl:.4f} | "
            f"Time/epoch: {epoch_time:.2f}s"
        )

    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time:.2f}s " f"({total_time / epochs:.2f}s/epoch avg)")
    epoch_times = torch.tensor(epoch_times)
    print(f"Mean time/epoch: {epoch_times.mean().item():.2f} ± {epoch_times.std(unbiased=False).item():.2f} s")

    return {
        "focus_metric": nll_history, # for plotting
        "epoch_times": epoch_times,
    }

