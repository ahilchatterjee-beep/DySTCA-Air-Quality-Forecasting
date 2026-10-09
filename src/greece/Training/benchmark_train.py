# ==========================================================
# LIGHTWEIGHT CPU TIMING TEST
#
# IMPORTANT:
# - Uses the EXACT SAME MODEL
# - Uses the EXACT SAME DATA
# - Does NOT modify architecture
# - Does NOT run training epochs
# - Does NOT use torch.profiler
# - Runs ONE batch only
# ==========================================================

import time
import torch




print("\n" + "=" * 70)
print("LIGHTWEIGHT CPU PERFORMANCE TEST")
print("=" * 70)

# ----------------------------------------------------------
# Force CPU
# ----------------------------------------------------------

device = torch.device("cpu")

model = model.to(device)
model.train()

# ----------------------------------------------------------
# Get ONE batch
# ----------------------------------------------------------

start = time.perf_counter()

X_batch, Y_batch, station_records = next(
    iter(train_loader)
)

X_batch = X_batch.to(device)
Y_batch = Y_batch.to(device)

data_time = time.perf_counter() - start

print("\nBatch information:")
print("  X shape :", tuple(X_batch.shape))
print("  Y shape :", tuple(Y_batch.shape))
print("  Device  :", X_batch.device)
print(f"  Data loading + transfer: {data_time:.4f} sec")


# ----------------------------------------------------------
# Same optimizer as training
# ----------------------------------------------------------

benchmark_optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

benchmark_optimizer.zero_grad(set_to_none=True)


# ==========================================================
# FORWARD
# ==========================================================

print("\nRunning forward pass...")

start = time.perf_counter()

output = model(
    X_batch,
    station_records
)

forward_time = time.perf_counter() - start

print(f"Forward time: {forward_time:.4f} sec")


# ==========================================================
# LOSS
# ==========================================================

print("Calculating loss...")

start = time.perf_counter()

loss = criterion(
    output,
    Y_batch
)

loss_time = time.perf_counter() - start

print(f"Loss time: {loss_time:.6f} sec")


# ==========================================================
# BACKWARD
# ==========================================================

print("Running backward pass...")

start = time.perf_counter()

loss.backward()

backward_time = time.perf_counter() - start

print(f"Backward time: {backward_time:.4f} sec")


# ==========================================================
# GRADIENT CLIPPING
# ==========================================================

print("Gradient clipping...")

start = time.perf_counter()

if GRADIENT_CLIPPING:

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        MAX_GRAD_NORM
    )

else:

    gradient_norm = 0.0

gradient_time = time.perf_counter() - start

print(f"Gradient clipping time: {gradient_time:.6f} sec")


# ==========================================================
# OPTIMIZER
# ==========================================================

print("Optimizer step...")

start = time.perf_counter()

benchmark_optimizer.step()

optimizer_time = time.perf_counter() - start

print(f"Optimizer time: {optimizer_time:.6f} sec")


# ==========================================================
# TOTAL
# ==========================================================

total_time = (
    data_time
    + forward_time
    + loss_time
    + backward_time
    + gradient_time
    + optimizer_time
)


print("\n" + "=" * 70)
print("CPU TIMING SUMMARY")
print("=" * 70)

print(f"Data loading       : {data_time:.4f} sec")
print(f"Forward            : {forward_time:.4f} sec")
print(f"Loss               : {loss_time:.6f} sec")
print(f"Backward           : {backward_time:.4f} sec")
print(f"Gradient clipping  : {gradient_time:.6f} sec")
print(f"Optimizer          : {optimizer_time:.6f} sec")

print("-" * 70)

print(f"TOTAL              : {total_time:.4f} sec")

print("=" * 70)


# ==========================================================
# NUMERICAL CHECK
# ==========================================================

print("\nNumerical check:")

print(
    "Output shape:",
    tuple(output.shape)
)

print(
    "NaNs:",
    torch.isnan(output).sum().item()
)

print(
    "Infs:",
    torch.isinf(output).sum().item()
)

print(
    "Loss:",
    float(loss.detach())
)

print(
    "Gradient norm:",
    float(
        gradient_norm
        if not torch.is_tensor(gradient_norm)
        else gradient_norm.detach()
    )
)

print("\nCPU timing test complete.")