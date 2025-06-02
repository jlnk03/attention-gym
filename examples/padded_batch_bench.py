"""Benchmark for padded batch processing with attention masking"""

import torch
import torch.nn.functional as F
from tabulate import tabulate
from typing import List, Tuple
from functools import lru_cache
import random
import time

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

# Configuration
data_type = torch.float16
BATCH_SIZE = 16  # Fixed batch size


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    """Calculate TFLOPS (trillion floating point operations per second)."""
    return multiplier * flops * (1e3 / time_ms) / 1e12


def print_header(text):
    """Print a nicely formatted header."""
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")


def create_padded_batch_data(
    min_seq_len: int,
    max_seq_len: int,
    head_dim: int,
    num_heads: int,
    device: str = "cuda",
    dtype=torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], int, int]:
    """
    Create a padded batch with random sequence lengths using functional padding approach.
    
    Args:
        min_seq_len/max_seq_len: Range for random sequence lengths
        head_dim: Dimension of each attention head
        num_heads: Number of attention heads
        device: Device to create tensors on
        dtype: Data type for the tensors
        
    Returns:
        q: Query tensor with padding [batch_size, num_heads, max_len, head_dim]
        k: Key tensor with padding [batch_size, num_heads, max_len, head_dim]
        v: Value tensor with padding [batch_size, num_heads, max_len, head_dim]
        attn_mask: Attention mask for padding [batch_size, num_heads, max_len, max_len]
        seq_lengths: Actual sequence length for each sample in batch
        max_len: Maximum sequence length in the batch
    """
    # Generate random sequence lengths for each sample in batch
    seq_lengths = [random.randint(min_seq_len, max_seq_len) for _ in range(BATCH_SIZE)]
    max_len = max(seq_lengths)
    
    # Create separate tensors for each sequence with their actual lengths
    q_list = []
    k_list = []
    v_list = []
    seq_masks = []
    
    for length in seq_lengths:
        # Create tensors for each sequence with its actual length
        q_seq = torch.randn(num_heads, length, head_dim, device=device, dtype=dtype, requires_grad=True)
        k_seq = torch.randn(num_heads, length, head_dim, device=device, dtype=dtype, requires_grad=True)
        v_seq = torch.randn(num_heads, length, head_dim, device=device, dtype=dtype, requires_grad=True)
        
        # Create mask for this sequence (1 = real token, 0 = padding)
        seq_mask = torch.ones(length, device=device, dtype=torch.bool)
        
        # Pad to max length
        q_padded = F.pad(q_seq, (0, 0, 0, max_len - length), value=0)
        k_padded = F.pad(k_seq, (0, 0, 0, max_len - length), value=0)
        v_padded = F.pad(v_seq, (0, 0, 0, max_len - length), value=0)
        mask_padded = F.pad(seq_mask, (0, max_len - length), value=0)
        
        q_list.append(q_padded)
        k_list.append(k_padded)
        v_list.append(v_padded)
        seq_masks.append(mask_padded)
    
    # Stack all padded sequences into a batch
    q = torch.stack(q_list, dim=0)  # [B, H, max_len, D]
    k = torch.stack(k_list, dim=0)
    v = torch.stack(v_list, dim=0)
    seq_masks = torch.stack(seq_masks, dim=0)  # [B, max_len]
    
    # Create attention mask where each token attends to all valid tokens
    # Start with a mask of all False (masked)
    attn_mask = torch.zeros(BATCH_SIZE, num_heads, max_len, max_len, device=device, dtype=torch.bool)
    
    # Set valid attention positions based on sequence masks
    for b in range(BATCH_SIZE):
        # Expand mask to 2D attention pattern (each token can attend to all valid tokens)
        valid_tokens = seq_masks[b]
        # Outer product of valid token masks to create 2D attention mask
        valid_attention = valid_tokens.unsqueeze(0) & valid_tokens.unsqueeze(1)
        attn_mask[b, :, :, :] = valid_attention
    
    # Convert to float mask for SDPA (0 = attend, -inf = mask)
    # Use the same dtype as query tensor
    float_mask = torch.zeros_like(attn_mask, dtype=dtype)
    float_mask.masked_fill_(~attn_mask, float("-inf"))
    
    return q, k, v, float_mask, seq_lengths, max_len, num_heads


def create_causal_padding_mask(
    seq_masks: torch.Tensor,
    seq_lengths: List[int],
    max_len: int,
    num_heads: int,
    device: str = "cuda",
    dtype=torch.float16,
) -> torch.Tensor:
    """
    Create a causal padding mask for attention.
    
    Args:
        seq_masks: Boolean mask indicating valid tokens [batch_size, max_len]
        seq_lengths: Actual sequence length for each sample in batch
        max_len: Maximum sequence length in the batch
        num_heads: Number of attention heads
        device: Device to create tensors on
        dtype: Data type for the output mask
        
    Returns:
        attn_mask: Causal attention mask with padding [batch_size, num_heads, max_len, max_len]
    """
    # Create causal mask (upper triangular part is masked)
    causal_mask = torch.triu(
        torch.ones(max_len, max_len, device=device), diagonal=1
    ).bool()
    
    # Create padding mask (True = attend, False = mask)
    attn_mask = torch.zeros(BATCH_SIZE, num_heads, max_len, max_len, device=device, dtype=torch.bool)
    
    for b in range(BATCH_SIZE):
        # Get valid token mask for this sequence
        valid_tokens = seq_masks[b]
        
        # Expand to 2D attention pattern with causality
        # Outer product of valid token masks AND causal constraint
        valid_attention = valid_tokens.unsqueeze(0) & valid_tokens.unsqueeze(1) & (~causal_mask[:max_len, :max_len])
        attn_mask[b, :, :, :] = valid_attention
    
    # Convert to float mask for SDPA (0 = attend, -inf = mask)
    # Use the same dtype as query tensor
    float_mask = torch.zeros_like(attn_mask, dtype=dtype)
    float_mask.masked_fill_(~attn_mask, float("-inf"))
    
    return float_mask


def create_flex_padding_block_mask(
    seq_masks: torch.Tensor,
    seq_lengths: List[int],
    max_len: int,
    num_heads: int,
    device: str = "cuda",
    causal: bool = False,
) -> torch.Tensor:
    """
    Create a block mask for FlexAttention with padding.
    
    Args:
        seq_masks: Boolean mask indicating valid tokens [batch_size, max_len]
        seq_lengths: Actual sequence length for each sample
        max_len: Maximum sequence length
        num_heads: Number of attention heads
        device: Device to create tensors on
        causal: Whether to use causal masking
        
    Returns:
        block_mask: Block mask for FlexAttention
    """
    def mask_mod(b, h, q_idx, kv_idx):
        # Get actual position within the sequence
        q_pos = q_idx % max_len
        kv_pos = kv_idx % max_len
        
        # Check if both positions are valid tokens using sequence masks
        # Use tensor operations instead of logical operators
        valid = seq_masks[b][q_pos] & seq_masks[b][kv_pos]
        
        # Apply causality constraint if requested
        if causal:
            causal_valid = q_pos >= kv_pos
            valid = valid & causal_valid
            
        return valid
    
    # Create the block mask
    block_mask = create_block_mask(mask_mod, BATCH_SIZE, num_heads, max_len, max_len, device=device)
    return block_mask


def run_padded_batch_benchmark(
    min_seq_len: int = 64,
    max_seq_len: int = 512,
    H: int = 16,
    D: int = 64,
    device: str = "cuda",
    causal: bool = False,
    check_correctness: bool = True,
):
    """
    Benchmark padded batch processing with different attention implementations.
    
    Args:
        min_seq_len/max_seq_len: Range for random sequence lengths
        H: Number of heads
        D: Head dimension
        device: Device to run on
        causal: Whether to use causal masking
        check_correctness: Whether to verify that implementations produce same results
    """
    # Create padded batch data
    q, k, v, dense_mask, seq_lengths, max_len, num_heads = create_padded_batch_data(
        min_seq_len, max_seq_len, D, H, device, data_type
    )
    
    # Get sequence masks
    seq_masks = torch.zeros(BATCH_SIZE, max_len, device=device, dtype=torch.bool)
    for b, length in enumerate(seq_lengths):
        seq_masks[b, :length] = True
    
    # For causal masking, create a different mask
    if causal:
        dense_mask = create_causal_padding_mask(
            seq_masks, seq_lengths, max_len, num_heads, device, data_type
        )
    
    # Create block mask for FlexAttention
    block_mask = create_flex_padding_block_mask(
        seq_masks, seq_lengths, max_len, num_heads, device, causal
    )
    
    grad_out = torch.randn_like(q)
    
    # Print sparsity info
    print(f"Block mask sparsity: {block_mask.sparsity():.2f}%")
    print(f"Batch size: {BATCH_SIZE}, Max sequence length: {max_len}")
    print(f"Sequence lengths: {seq_lengths}")
    
    # Define attention operations
    sdpa_mask = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask)
    flex = lambda: flex_attention(q, k, v, block_mask=block_mask)
    
    # Calculate FLOPs for the entire padded batch
    # Includes padding which is wasteful but standard practice
    full_flops = BATCH_SIZE * H * D * max_len * max_len
    
    # Calculate effective FLOPs excluding padding
    effective_tokens = sum(seq_lengths)
    effective_flops = BATCH_SIZE * H * D * effective_tokens * effective_tokens
    
    # Calculate sparse FLOPs for FlexAttention (only compute non-masked elements)
    density = (100 - block_mask.sparsity()) / 100
    sparse_flops = density * BATCH_SIZE * H * D * max_len * max_len
    
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
        f"{calculate_tflops(effective_flops, fw_time, 4):.2f}",
        f"{bw_time:.4f}",
        f"{calculate_tflops(full_flops, bw_time, 10):.2f}",
        f"{calculate_tflops(effective_flops, bw_time, 10):.2f}",
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
        "flexattention",
        f"{fw_time:.4f}",
        f"{calculate_tflops(sparse_flops, fw_time, 4):.2f}",
        f"{calculate_tflops(effective_flops, fw_time, 4):.2f}",
        f"{bw_time:.4f}",
        f"{calculate_tflops(sparse_flops, bw_time, 10):.2f}",
        f"{calculate_tflops(effective_flops, bw_time, 10):.2f}",
    ])
    
    # Clear memory
    del out
    q.grad = k.grad = v.grad = None
    torch.cuda.empty_cache()
    
    # Print results
    mask_type = "Causal + Padding" if causal else "Padding"
    print_header(f"Padded Batch Attention Benchmark - {mask_type} Mask (B={BATCH_SIZE})")
    print(
        tabulate(
            rows,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW Padded TF/s",
                "FW Effective TF/s",
                "BW Time (ms)",
                "BW Padded TF/s",
                "BW Effective TF/s",
            ],
            tablefmt="grid",
        )
    )
    
    # Print efficiency details
    padded_size = BATCH_SIZE * max_len * max_len
    actual_size = sum(length * length for length in seq_lengths)
    padding_efficiency = (actual_size / padded_size) * 100
    
    print(f"\nPadding efficiency: {padding_efficiency:.2f}%")
    print(f"Total tokens (with padding): {BATCH_SIZE * max_len}")
    print(f"Effective tokens (no padding): {effective_tokens}")
    print(f"Input dimensions: B={BATCH_SIZE}, H={H}, D={D}")


def main(
    min_seq_len: int = 64,
    max_seq_len: int = 512,
    heads: int = 16,
    dim: int = 64,
    causal: bool = False,
    check_correctness: bool = True,
):
    """
    Run padded batch attention benchmark with configurable parameters.
    """
    run_padded_batch_benchmark(
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        H=heads,
        D=dim,
        causal=causal,
        check_correctness=check_correctness,
    )


if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser, CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    
    CLI(main) 