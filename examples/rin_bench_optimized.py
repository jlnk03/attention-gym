"""Optimized Benchmark for Retrieval-in-Network (RIN) Cross-Attention Mask with precomputed IDs"""

import torch
import torch.nn.functional as F
from tabulate import tabulate
from typing import List, Tuple
from functools import lru_cache
from itertools import accumulate
import random

from torch.nn.attention.flex_attention import (
    create_block_mask,
    create_mask,
    flex_attention,
)

from triton.testing import do_bench

# Set seed for reproducibility
random.seed(0)
torch.manual_seed(0)

# Configure PyTorch
torch.set_default_device("cuda")
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function for better performance
flex_attention = torch.compile(flex_attention, dynamic=False)
# Alternative option if above still fails:
# flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

# Configuration
data_type = torch.float16


@lru_cache
def create_block_mask_cached(mask_mod, B, H, M, N, device="cuda"):
    """Create and cache a block mask to avoid recomputation."""
    block_mask = create_block_mask(mask_mod, B, H, M, N, device=device)
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    """Calculate TFLOPS (trillion floating point operations per second)."""
    return multiplier * flops * (1e3 / time_ms) / 1e12


def print_header(text):
    """Print a nicely formatted header."""
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")


def precompute_image_ids(
    img_lengths: List[int], latent_len: int, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute image IDs for both query and key/value tokens.
    
    Returns:
        Tuple of (query_ids, kv_ids) tensors
    """
    n_images = len(img_lengths)
    
    # For queries: each image has exactly latent_len queries
    total_q = n_images * latent_len
    query_ids = torch.arange(n_images, device=device).repeat_interleave(latent_len)
    
    # For key/value tokens: each image has img_lengths[i] tokens
    total_kv = sum(img_lengths)
    kv_ids = torch.zeros(total_kv, dtype=torch.long, device=device)
    
    offset = 0
    for i, length in enumerate(img_lengths):
        kv_ids[offset:offset+length] = i
        offset += length
    
    return query_ids, kv_ids


def generate_rin_mask_mod(
    img_lengths: List[int], latent_len: int, device: str = "cuda"
):
    """
    Generate optimized mask mod for RIN Cross-Attention using precomputed IDs.
    
    Each image i owns:
      • latent_len query tokens (Q side)
      • img_lengths[i] key/value tokens (KV side)
    Queries may only see KV tokens from the same image.
    """
    query_ids, kv_ids = precompute_image_ids(img_lengths, latent_len, device)
    total_kv = sum(img_lengths)
    
    def mask_mod(b, h, q_idx, kv_idx):
        # Look up precomputed IDs
        kv_idx = torch.clamp(kv_idx, 0, total_kv - 1)
        return query_ids[q_idx] == kv_ids[kv_idx]

    return mask_mod


def run_rin_benchmark(
    n_images: int = 16,
    latent_len: int = 128,
    min_tokens: int = 512,
    max_tokens: int = 512,
    B: int = 1,
    H: int = 16,
    D: int = 64,
    device: str = "cuda",
    check_correctness: bool = True,
):
    """
    Benchmark RIN Cross-Attention with different attention implementations.
    
    Args:
        n_images: Number of images to process
        latent_len: Number of latent tokens per image
        min_tokens/max_tokens: Range for random image token lengths
        B: Batch size (fixed to 1 for this benchmark)
        H: Number of heads
        D: Head dimension
        device: Device to run on
        check_correctness: Whether to verify that implementations produce same results
    """
    # Generate random image token lengths
    img_lengths = [random.randint(min_tokens, max_tokens) for _ in range(n_images)]
    
    # Calculate dimensions
    Q = n_images * latent_len  # Total query tokens
    KV = sum(img_lengths)      # Total key/value tokens
    
    # Generate the mask mod
    mask_mod = generate_rin_mask_mod(img_lengths, latent_len, device)
    
    # Create block mask for FlexAttention
    block_mask = create_block_mask_cached(mask_mod, B, H, Q, KV, device=device)
    
    # Create dense mask for SDPA
    dense_mask = create_mask(mask_mod, B, H, Q, KV, device=device)
    
    # Create input tensors
    q = torch.randn(B, H, Q, D, device=device, dtype=data_type, requires_grad=True)
    k = torch.randn(B, H, KV, D, device=device, dtype=data_type, requires_grad=True)
    v = torch.randn_like(k)
    grad_out = torch.randn_like(q)
    
    # Define attention operations
    sdpa_mask = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask)
    flex = lambda: flex_attention(q, k, v, block_mask=block_mask)
    
    # Calculate FLOPs
    density = (100 - block_mask.sparsity()) / 100
    full_flops = B * H * D * Q * KV
    sparse_flops = density * full_flops
    
    # Check correctness if requested
    if check_correctness:
        sdpa_out = sdpa_mask()
        flex_out = flex()
        try:
            torch.testing.assert_close(sdpa_out, flex_out, atol=1e-1, rtol=1e-2)
            print("Correctness check passed ✅")
        except AssertionError as e:
            print(f"Correctness check failed: {e}")
        
        # Clear memory
        del sdpa_out, flex_out
        q.grad = k.grad = v.grad = None
        torch.cuda.empty_cache()
    
    # Benchmark forward and backward passes
    rows = []
    
    # Benchmark SDPA with mask
    fw_time = do_bench(sdpa_mask)
    out = sdpa_mask()
    bw_time = do_bench(lambda: out.backward(grad_out, retain_graph=True))
    
    rows.append([
        "F.sdpa + mask",
        f"{fw_time:.4f}",
        f"{calculate_tflops(full_flops, fw_time, 4):.2f}",
        f"{bw_time:.4f}",
        f"{calculate_tflops(full_flops, bw_time, 10):.2f}",
    ])
    
    # Clear memory
    del out
    q.grad = k.grad = v.grad = None
    torch.cuda.empty_cache()
    
    # Benchmark FlexAttention
    fw_time = do_bench(flex)
    out = flex()
    bw_time = do_bench(lambda: out.backward(grad_out, retain_graph=True))
    
    rows.append([
        "flexattention (indexing)",
        f"{fw_time:.4f}",
        f"{calculate_tflops(sparse_flops, fw_time, 4):.2f}",
        f"{bw_time:.4f}",
        f"{calculate_tflops(sparse_flops, bw_time, 10):.2f}",
    ])
    
    # Clear memory
    del out
    q.grad = k.grad = v.grad = None
    torch.cuda.empty_cache()
    
    # Print results
    print_header(f"RIN Cross-Attention Benchmark (B={B}, n_images={n_images})")
    print(
        tabulate(
            rows,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ],
            tablefmt="grid",
        )
    )
    print(f"\nBlockMask sparsity: {block_mask.sparsity():.2f}%   (Q={Q}, KV={KV})")
    print(f"Input dimensions: B={B}, H={H}, D={D}")
    print(f"Image lengths: {img_lengths}")
    print(f"Latent length per image: {latent_len}")
    

def main(
    n_images: int = 64,
    latent_len: int = 128,
    min_tokens: int = 64,
    max_tokens: int = 512,
    heads: int = 16,
    dim: int = 64,
    check_correctness: bool = True,
):
    """
    Run RIN Cross-Attention benchmark with configurable parameters.
    """
    # Always run with batch size of 1 as requested
    run_rin_benchmark(
        n_images=n_images,
        latent_len=latent_len,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        B=1,  # Fixed batch size of 1
        H=heads,
        D=dim,
        check_correctness=check_correctness,
    )


if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser, CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    
    CLI(main) 