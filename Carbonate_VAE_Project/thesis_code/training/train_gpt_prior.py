"""
train_gpt_prior.py V2 — Train Micro-GPT (2D position-aware) as VQ-VAE Prior
=============================================================================
Key changes from V1:
- Micro-GPT: 4 layers, 4 heads, 256 dim, dropout=0.25
- 2D positional embeddings (row/col) in nanogpt.py
- 100 epochs with weight_decay=0.1
- Saves checkpoint for new config
"""

import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from thesis_code.config import PROJECT_ROOT


from thesis_code.models.nanogpt import GPT, GPTConfig

# ── Paths ──
CODES_PATH = os.path.join(PROJECT_ROOT, "results", "checkpoints", "latent_codes.npy")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CHECKPOINT_DIR, "gpt_prior_v2.pt")

# ── Micro-GPT Hyperparameters ──
VOCAB_SIZE = 512
BLOCK_SIZE = 1024       # 32 x 32 flattened
N_LAYER = 4             # Reduced from 6
N_HEAD = 4              # Reduced from 8
N_EMBD = 256            # Reduced from 384
DROPOUT = 0.25          # Aggressive dropout to prevent overfitting
BIAS = False

BATCH_SIZE = 16
EPOCHS = 100            # Increased from 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1      # Strong weight decay
BETAS = (0.9, 0.95)

WEIGHT_CLIP_MIN = 1.0
WEIGHT_CLIP_MAX = 5.0

GRAD_CLIP = 1.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[DEVICE] Using: {device}")


def load_and_prepare_data():
    codes = np.load(CODES_PATH).astype(np.int64)
    N, H, W = codes.shape
    print(f"[DATA] Loaded: {codes.shape}, range [{codes.min()}, {codes.max()}]")

    flat = codes.reshape(N, -1)  # (N, 1024)
    x = flat[:, :-1]   # (N, 1023)
    y = flat[:, 1:]    # (N, 1023)
    print(f"[DATA] x shape: {x.shape}, y shape: {y.shape}")

    np.random.seed(42)
    perm = np.random.permutation(N)
    split = int(N * 0.9)
    train_idx = perm[:split]
    val_idx = perm[split:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    print(f"[DATA] Train: {len(x_train)}, Val: {len(x_val)}")

    all_flat = flat.flatten()
    counts = np.bincount(all_flat, minlength=VOCAB_SIZE).astype(np.float32)
    total = counts.sum()
    freqs = counts / total
    weights = 1.0 / (freqs + 1e-5)
    weights = np.clip(weights, WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    active = (counts > 0).sum()
    print(f"[WEIGHTS] Active tokens: {active}/{VOCAB_SIZE}")
    print(f"[WEIGHTS] Freq range: [{freqs[counts > 0].min():.6f}, {freqs[counts > 0].max():.4f}]")
    print(f"[WEIGHTS] Weight range: [{weights.min():.2f}, {weights.max():.2f}]")

    # Also compute valid_tokens for future inference
    valid_tokens = np.where(counts > 0)[0].astype(np.int64)

    return (x_train, y_train), (x_val, y_val), class_weights, valid_tokens


def get_batch(x_data, y_data, batch_size):
    idx = torch.randint(0, len(x_data), (batch_size,))
    x = torch.tensor(x_data[idx], dtype=torch.long, device=device)
    y = torch.tensor(y_data[idx], dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def estimate_val_loss(model, x_val, y_val, class_weights, eval_iters=20):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(x_val, y_val, BATCH_SIZE)
        logits, loss = model(x, targets=y)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE), y.view(-1),
            weight=class_weights, ignore_index=-1
        )
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


def main():
    print("=" * 70)
    print("  Micro-GPT V2 Prior Training (2D Position-Aware)")
    print(f"  Config: {N_LAYER}L {N_HEAD}H {N_EMBD}D, vocab={VOCAB_SIZE}, block={BLOCK_SIZE}")
    print(f"  Dropout: {DROPOUT}, Weight Decay: {WEIGHT_DECAY}")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"  Weight clip: [{WEIGHT_CLIP_MIN}, {WEIGHT_CLIP_MAX}]")
    print("=" * 70)

    (x_train, y_train), (x_val, y_val), class_weights, valid_tokens = load_and_prepare_data()
    steps_per_epoch = len(x_train) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    print(f"[DATA] Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
    print(f"[DATA] Valid token count: {len(valid_tokens)} (saved for inference)")

    # Save valid_tokens for inference scripts
    np.save(os.path.join(CHECKPOINT_DIR, "valid_tokens.npy"), valid_tokens)

    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=BIAS,
    )
    model = GPT(config)
    model.to(device)
    model.train()

    optimizer = model.configure_optimizers(
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        betas=BETAS,
        device_type=device_type,
    )

    def cosine_lr(step):
        if step < total_steps // 10:
            return step / (total_steps // 10)
        progress = (step - total_steps // 10) / (total_steps - total_steps // 10)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr)

    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 20
    history = {'train': [], 'val': []}

    t_start = time.time()
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(len(x_train))
        x_train_shuf = x_train[perm]
        y_train_shuf = y_train[perm]

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            start = step * BATCH_SIZE
            end = start + BATCH_SIZE
            x = torch.tensor(x_train_shuf[start:end], dtype=torch.long, device=device)
            y = torch.tensor(y_train_shuf[start:end], dtype=torch.long, device=device)

            logits, _ = model(x, targets=y)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE), y.view(-1),
                weight=class_weights, ignore_index=-1
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

        avg_train = epoch_loss / steps_per_epoch
        avg_val = estimate_val_loss(model, x_val, y_val, class_weights)
        history['train'].append(avg_train)
        history['val'].append(avg_val)

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"train: {avg_train:.4f} | val: {avg_val:.4f} | "
              f"lr: {lr_now:.2e} | time: {elapsed:.0f}s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'history': history,
                'class_weights': class_weights.cpu(),
                'valid_tokens': torch.tensor(valid_tokens, dtype=torch.long),
            }, CKPT_PATH)
            print(f"    [BEST] Saved checkpoint (val={best_val_loss:.4f})")
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            periodic_path = os.path.join(CHECKPOINT_DIR, f"gpt_prior_v2_ep{epoch:03d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'history': history,
                'class_weights': class_weights.cpu(),
                'valid_tokens': torch.tensor(valid_tokens, dtype=torch.long),
            }, periodic_path)

        if patience_counter >= PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs. Stopping.")
            break

    elapsed_total = time.time() - t_start
    print(f"\n[DONE] Training complete in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"[BEST] Validation loss: {best_val_loss:.4f}")
    print(f"[SAVE] Final model: {CKPT_PATH}")

    np.savez(os.path.join(CHECKPOINT_DIR, "gpt_prior_v2_history.npz"),
             train=history['train'], val=history['val'])


if __name__ == "__main__":
    main()
