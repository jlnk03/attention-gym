"""Combined benchmark for RIN with padding vs RIN with one batch approaches"""

import torch
import torch.nn.functional as F
from tabulate import tabulate
from typing import List, Tuple, Dict
from functools import lru_cache
from itertools import accumulate
import random
import time
import os
import json
import csv
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

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
sdpa = torch.compile(F.scaled_dot_product_attention, dynamic=False)

# Configuration
data_type = torch.float16


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    """Calculate TFLOPS (trillion floating point operations per second)."""
    return multiplier * flops * (1e3 / time_ms) / 1e12


def print_header(text, file=None):
    """Print a nicely formatted header."""
    width = 91
    header = "╔" + "═" * (width - 2) + "╗"
    title = f"║ {text.center(width - 4)} ║"
    footer = "╚" + "═" * (width - 2) + "╝"
    
    print(header)
    print(title)
    print(footer)
    
    if file:
        file.write(f"\n## {text}\n\n")


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


def generate_shared_data(
    n_images: int,
    latent_len: int,
    min_tokens: int,
    max_tokens: int,
    H: int,
    D: int,
    device: str = "cuda",
    dtype=torch.float16,
    causal: bool = False,
) -> Tuple:
    """
    Generate data for both benchmarks to ensure fair comparison.
    
    Args:
        n_images: Number of images (equivalent to batch size for padded approach)
        latent_len: Number of latent tokens per image
        min_tokens/max_tokens: Range for random image token lengths
        H: Number of heads
        D: Head dimension
        device: Device to create tensors on
        dtype: Data type for the tensors
        causal: Whether to use causal masking
        
    Returns:
        Data structures for both benchmarking approaches
    """
    # Generate random image token lengths - SHARED between approaches
    img_lengths = [random.randint(min_tokens, max_tokens) for _ in range(n_images)]
    max_len = max(img_lengths)
    
    print(f"Generated random sequence lengths: {img_lengths}")
    print(f"Maximum length: {max_len}")
    
    # ======== PADDED BATCH DATA ========
    # Create padded batch data (each image = one batch item)
    # For cross-attention: Q has latent_len tokens, K/V have img_lengths[i] tokens
    padded_q_list = []
    padded_k_list = []
    padded_v_list = []
    kv_masks = []
    
    for length in img_lengths:
        # Create query tensors: latent_len tokens per image (cross-attention queries)
        q_seq = torch.randn(H, latent_len, D, device=device, dtype=dtype, requires_grad=True)
        
        # Create key/value tensors: img_lengths[i] tokens per image (cross-attention keys/values)
        k_seq = torch.randn(H, length, D, device=device, dtype=dtype, requires_grad=True)
        v_seq = torch.randn(H, length, D, device=device, dtype=dtype, requires_grad=True)
        
        # Create mask for key/value sequence (1 = real token, 0 = padding)
        kv_mask = torch.ones(length, device=device, dtype=torch.bool)
        
        # Pad key/value to max length (queries don't need padding as they're fixed size)
        k_padded = F.pad(k_seq, (0, 0, 0, max_len - length), value=0)
        v_padded = F.pad(v_seq, (0, 0, 0, max_len - length), value=0)
        kv_mask_padded = F.pad(kv_mask, (0, max_len - length), value=0)
        
        padded_q_list.append(q_seq)  # No padding needed for queries
        padded_k_list.append(k_padded)
        padded_v_list.append(v_padded)
        kv_masks.append(kv_mask_padded)
    
    # Stack all padded sequences into a batch
    padded_q = torch.stack(padded_q_list, dim=0)  # [B, H, latent_len, D]
    padded_k = torch.stack(padded_k_list, dim=0)  # [B, H, max_len, D]
    padded_v = torch.stack(padded_v_list, dim=0)  # [B, H, max_len, D]
    kv_masks = torch.stack(kv_masks, dim=0)  # [B, max_len]

    print(f"Padded Q shape: {padded_q.shape}")
    print(f"Padded K shape: {padded_k.shape}")
    print(f"Padded V shape: {padded_v.shape}")
    print(f"KV masks shape: {kv_masks.shape}")
    
    # Create cross-attention mask for padding
    # Shape: [B, H, latent_len, max_len] (queries can attend to valid key/value tokens)
    padded_attn_mask = torch.zeros(n_images, H, latent_len, max_len, device=device, dtype=torch.bool)
    
    # Set valid attention positions: all queries can attend to all valid key/value tokens within same image
    for b in range(n_images):
        # All latent queries in this batch item can attend to valid key/value tokens
        valid_kv_tokens = kv_masks[b]  # [max_len]
        # Expand to [latent_len, max_len] - all queries can attend to valid KV tokens
        padded_attn_mask[b, :, :, :] = valid_kv_tokens.unsqueeze(0).expand(latent_len, -1)
    
    # Convert to float mask for SDPA (0 = attend, -inf = mask)
    padded_float_mask = torch.zeros_like(padded_attn_mask, dtype=dtype)
    padded_float_mask.masked_fill_(~padded_attn_mask, float("-inf"))
    
    # Create padded block mask for FlexAttention
    def padded_mask_mod(b, h, q_idx, kv_idx):
        # For cross-attention: q_idx is in [0, latent_len), kv_idx is in [0, max_len)
        # Check if the key/value token is valid (not padding)
        return kv_masks[b][kv_idx]
    
    padded_block_mask = create_block_mask(padded_mask_mod, n_images, H, latent_len, max_len, device=device)
    
    padded_data = {
        "q": padded_q,
        "k": padded_k,
        "v": padded_v,
        "dense_mask": padded_float_mask,
        "block_mask": padded_block_mask,
        "seq_lengths": img_lengths,
        "max_len": max_len,
    }
    
    # ======== RIN DATA (CONCATENATED) ========
    # Create RIN data (all images concatenated into a single batch)
    
    # For RIN, we have:
    # - Latent queries: n_images * latent_len
    # - KV tokens: sum of all img_lengths
    
    Q = n_images * latent_len  # Total query tokens
    KV = sum(img_lengths)      # Total key/value tokens
    
    # Create input tensors for RIN
    rin_q = torch.randn(1, H, Q, D, device=device, dtype=dtype, requires_grad=True)
    rin_k = torch.randn(1, H, KV, D, device=device, dtype=dtype, requires_grad=True)
    rin_v = torch.randn_like(rin_k)

    print(f"RIN Q shape: {rin_q.shape}")
    print(f"RIN K shape: {rin_k.shape}")
    print(f"RIN V shape: {rin_v.shape}")
    
    # Generate the RIN mask mod using precomputed IDs for efficiency
    query_ids, kv_ids = precompute_image_ids(img_lengths, latent_len, device)
    total_kv = sum(img_lengths)

    def rin_mask_mod(b, h, q_idx, kv_idx):
        # Look up precomputed IDs
        kv_idx = torch.clamp(kv_idx, 0, total_kv - 1)
        return query_ids[q_idx] == kv_ids[kv_idx]

    # Create block mask for FlexAttention
    rin_block_mask = create_block_mask(rin_mask_mod, 1, H, Q, KV, device=device)
    # Create dense mask for SDPA
    rin_dense_mask = create_mask(rin_mask_mod, 1, H, Q, KV, device=device)
    
    rin_data = {
        "q": rin_q,
        "k": rin_k,
        "v": rin_v,
        "dense_mask": rin_dense_mask,
        "block_mask": rin_block_mask,
        "img_lengths": img_lengths,
        "latent_len": latent_len,
        "Q": Q,
        "KV": KV,
    }
    
    return padded_data, rin_data


def run_combined_benchmark(
    n_images: int = 16,
    latent_len: int = 128,
    min_tokens: int = 256,
    max_tokens: int = 256,
    H: int = 16,
    D: int = 64,
    device: str = "cuda",
    causal: bool = False,
    check_correctness: bool = True,
    export_dir: str = "results",
):
    """
    Run both benchmarks with comparable settings and data.
    
    Args:
        n_images: Number of images (= batch size for padded approach)
        latent_len: Number of latent tokens per image (for RIN)
        min_tokens/max_tokens: Range for random image token lengths
        H: Number of heads
        D: Head dimension
        device: Device to run on
        causal: Whether to use causal masking
        check_correctness: Whether to verify that implementations produce same results
        export_dir: Directory to export results
    """
    # Create export directory if it doesn't exist
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Create output files with generic names
    results_file = export_path / "benchmark_results.md"
    csv_file = export_path / "benchmark_results.csv"
    json_file = export_path / "benchmark_results.json"
    plot_file = export_path / "benchmark_plot.png"
    
    # Open markdown file for writing
    with open(results_file, "w") as md_file:
        md_file.write("# RIN with Padding vs One Batch Benchmark\n\n")
        md_file.write(f"**Run configuration:**\n\n")
        md_file.write(f"- Number of images: {n_images}\n")
        md_file.write(f"- Latent tokens per image: {latent_len}\n")
        md_file.write(f"- Min/Max tokens: {min_tokens}/{max_tokens}\n")
        md_file.write(f"- Heads: {H}\n")
        md_file.write(f"- Dimension: {D}\n")
        md_file.write(f"- Causal: {causal}\n\n")
        
        # Generate shared data for both benchmarks
        padded_data, rin_data = generate_shared_data(
            n_images, latent_len, min_tokens, max_tokens, H, D, device, data_type, causal
        )
        
        # Record all benchmark results in a dictionary to export as JSON later
        all_results = {
            "config": {
                "n_images": n_images,
                "latent_len": latent_len,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
                "heads": H,
                "dim": D,
                "causal": causal,
            },
            "data": {
                "seq_lengths": padded_data["seq_lengths"],
                "max_len": padded_data["max_len"],
                "total_kv": rin_data["KV"],
            },
            "padding": {},
            "one_batch": {},
            "comparison": {},
        }
        
        print_header("Data Generation Summary", md_file)
        
        # Data summary
        data_summary = [
            ["Number of images", f"{n_images}"],
            ["Latent tokens per image", f"{latent_len}"],
            ["Total query tokens (Cross-Attention)", f"{rin_data['Q']}"],
            ["Total KV tokens (Cross-Attention)", f"{rin_data['KV']}"],
            ["Query tokens per batch item (Padding)", f"{latent_len}"],
            ["Max KV sequence length (Padding)", f"{padded_data['max_len']}"],
            ["Padding block mask sparsity", f"{padded_data['block_mask'].sparsity():.2f}%"],
            ["One Batch block mask sparsity", f"{rin_data['block_mask'].sparsity():.2f}%"],
        ]
        
        print(tabulate(data_summary, tablefmt="simple"))
        md_file.write(tabulate(data_summary, headers=["Metric", "Value"], tablefmt="pipe") + "\n\n")
        
        # Save data details to results dict
        all_results["data"]["padding_sparsity"] = float(f"{padded_data['block_mask'].sparsity():.4f}")
        all_results["data"]["one_batch_sparsity"] = float(f"{rin_data['block_mask'].sparsity():.4f}")
        
        # Print summary to console
        for row in data_summary:
            print(f"{row[0]}: {row[1]}")
        
        # ======== BENCHMARK PADDED APPROACH ========
        print_header("RIN with Padding (each image = one batch item)", md_file)
        
        # Extract padded data
        p_q = padded_data["q"]
        p_k = padded_data["k"]
        p_v = padded_data["v"]
        p_dense_mask = padded_data["dense_mask"]
        p_block_mask = padded_data["block_mask"]
        p_grad_out = torch.randn_like(p_q)
        
        # Define padded attention operations
        p_sdpa_mask = lambda: sdpa(p_q, p_k, p_v, attn_mask=p_dense_mask)
        p_flex = lambda: flex_attention(p_q, p_k, p_v, block_mask=p_block_mask)
        
        # Calculate FLOPs for padded approach (cross-attention)
        p_max_len = padded_data["max_len"]
        p_full_flops = n_images * H * D * latent_len * p_max_len  # Cross-attention: Q_len * KV_len
        p_density = (100 - p_block_mask.sparsity()) / 100
        p_sparse_flops = p_density * p_full_flops
        
        # Check correctness if requested
        if check_correctness:
            p_sdpa_out = p_sdpa_mask()
            p_flex_out = p_flex()
            try:
                torch.testing.assert_close(p_sdpa_out, p_flex_out, atol=1e-1, rtol=1e-2)
                print("Padding approach correctness check passed ✅")
                md_file.write("Padding approach correctness check passed ✅\n\n")
            except AssertionError as e:
                print(f"Padding approach correctness check failed: {e}")
                md_file.write(f"Padding approach correctness check failed: {e}\n\n")
            
            # Clear memory
            del p_sdpa_out, p_flex_out
            p_q.grad = p_k.grad = p_v.grad = None
            torch.cuda.empty_cache()
        
        # Benchmark padded forward and backward passes
        p_rows = []
        
        # Benchmark SDPA with mask
        p_fw_time = do_bench(p_sdpa_mask)
        p_out = p_sdpa_mask()
        p_bw_time = do_bench(lambda: p_out.backward(p_grad_out, retain_graph=True))
        
        p_rows.append([
            "F.sdpa + mask",
            f"{p_fw_time:.4f}",
            f"{calculate_tflops(p_full_flops, p_fw_time, 4):.2f}",
            f"{p_bw_time:.4f}",
            f"{calculate_tflops(p_full_flops, p_bw_time, 10):.2f}",
        ])
        
        # Save results to dictionary
        all_results["padding"]["sdpa"] = {
            "fw_time": float(f"{p_fw_time:.4f}"),
            "fw_tflops": float(f"{calculate_tflops(p_full_flops, p_fw_time, 4):.4f}"),
            "bw_time": float(f"{p_bw_time:.4f}"),
            "bw_tflops": float(f"{calculate_tflops(p_full_flops, p_bw_time, 10):.4f}"),
        }
        
        # Clear memory
        del p_out
        p_q.grad = p_k.grad = p_v.grad = None
        torch.cuda.empty_cache()
        
        # Benchmark FlexAttention
        p_fw_time = do_bench(p_flex)
        p_out = p_flex()
        p_bw_time = do_bench(lambda: p_out.backward(p_grad_out, retain_graph=True))
        
        p_rows.append([
            "flexattention (indexing)",
            f"{p_fw_time:.4f}",
            f"{calculate_tflops(p_sparse_flops, p_fw_time, 4):.2f}",
            f"{p_bw_time:.4f}",
            f"{calculate_tflops(p_sparse_flops, p_bw_time, 10):.2f}",
        ])
        
        # Save results to dictionary
        all_results["padding"]["flex"] = {
            "fw_time": float(f"{p_fw_time:.4f}"),
            "fw_tflops": float(f"{calculate_tflops(p_sparse_flops, p_fw_time, 4):.4f}"),
            "bw_time": float(f"{p_bw_time:.4f}"),
            "bw_tflops": float(f"{calculate_tflops(p_sparse_flops, p_bw_time, 10):.4f}"),
        }
        
        # Clear memory
        del p_out
        p_q.grad = p_k.grad = p_v.grad = None
        torch.cuda.empty_cache()
        
        # Print padded results
        padded_table = tabulate(
            p_rows,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ],
            tablefmt="grid",
        )
        print(padded_table)
        
        # Write to markdown with pipe table format
        md_file.write(tabulate(
            p_rows,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ],
            tablefmt="pipe",
        ) + "\n\n")
        
        # ======== BENCHMARK RIN APPROACH ========
        print_header("RIN with One Batch (all images concatenated)", md_file)
        
        # Extract RIN data
        r_q = rin_data["q"]
        r_k = rin_data["k"]
        r_v = rin_data["v"]
        r_dense_mask = rin_data["dense_mask"]
        r_block_mask = rin_data["block_mask"]
        r_grad_out = torch.randn_like(r_q)
        
        # Define RIN attention operations
        r_sdpa_mask = lambda: sdpa(r_q, r_k, r_v, attn_mask=r_dense_mask)
        r_flex = lambda: flex_attention(r_q, r_k, r_v, block_mask=r_block_mask)
        
        # Calculate FLOPs for RIN approach
        r_Q = rin_data["Q"]
        r_KV = rin_data["KV"]
        r_full_flops = 1 * H * D * r_Q * r_KV  # B=1 for RIN
        r_density = (100 - r_block_mask.sparsity()) / 100
        r_sparse_flops = r_density * r_full_flops
        
        # Check correctness if requested
        if check_correctness:
            r_sdpa_out = r_sdpa_mask()
            r_flex_out = r_flex()
            try:
                torch.testing.assert_close(r_sdpa_out, r_flex_out, atol=1e-1, rtol=1e-2)
                print("One Batch approach correctness check passed ✅")
                md_file.write("One Batch approach correctness check passed ✅\n\n")
            except AssertionError as e:
                print(f"One Batch approach correctness check failed: {e}")
                md_file.write(f"One Batch approach correctness check failed: {e}\n\n")
            
            # Clear memory
            del r_sdpa_out, r_flex_out
            r_q.grad = r_k.grad = r_v.grad = None
            torch.cuda.empty_cache()
        
        # Benchmark RIN forward and backward passes
        r_rows = []
        
        # Benchmark SDPA with mask
        r_fw_time = do_bench(r_sdpa_mask)
        r_out = r_sdpa_mask()
        r_bw_time = do_bench(lambda: r_out.backward(r_grad_out, retain_graph=True))
        
        r_rows.append([
            "F.sdpa + mask",
            f"{r_fw_time:.4f}",
            f"{calculate_tflops(r_full_flops, r_fw_time, 4):.2f}",
            f"{r_bw_time:.4f}",
            f"{calculate_tflops(r_full_flops, r_bw_time, 10):.2f}",
        ])
        
        # Save results to dictionary
        all_results["one_batch"]["sdpa"] = {
            "fw_time": float(f"{r_fw_time:.4f}"),
            "fw_tflops": float(f"{calculate_tflops(r_full_flops, r_fw_time, 4):.4f}"),
            "bw_time": float(f"{r_bw_time:.4f}"),
            "bw_tflops": float(f"{calculate_tflops(r_full_flops, r_bw_time, 10):.4f}"),
        }
        
        # Clear memory
        del r_out
        r_q.grad = r_k.grad = r_v.grad = None
        torch.cuda.empty_cache()
        
        # Benchmark FlexAttention
        r_fw_time = do_bench(r_flex)
        r_out = r_flex()
        r_bw_time = do_bench(lambda: r_out.backward(r_grad_out, retain_graph=True))
        
        r_rows.append([
            "flexattention (indexing)",
            f"{r_fw_time:.4f}",
            f"{calculate_tflops(r_sparse_flops, r_fw_time, 4):.2f}",
            f"{r_bw_time:.4f}",
            f"{calculate_tflops(r_sparse_flops, r_bw_time, 10):.2f}",
        ])
        
        # Save results to dictionary
        all_results["one_batch"]["flex"] = {
            "fw_time": float(f"{r_fw_time:.4f}"),
            "fw_tflops": float(f"{calculate_tflops(r_sparse_flops, r_fw_time, 4):.4f}"),
            "bw_time": float(f"{r_bw_time:.4f}"),
            "bw_tflops": float(f"{calculate_tflops(r_sparse_flops, r_bw_time, 10):.4f}"),
        }
        
        # Clear memory
        del r_out
        r_q.grad = r_k.grad = r_v.grad = None
        torch.cuda.empty_cache()
        
        # Print RIN results
        rin_table = tabulate(
            r_rows,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ],
            tablefmt="grid",
        )
        print(rin_table)
        
        # Write to markdown with pipe table format
        md_file.write(tabulate(
            r_rows,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ],
            tablefmt="pipe",
        ) + "\n\n")
        
        # ======== COMPARATIVE ANALYSIS ========
        print_header("Padding vs. One Batch Comparison", md_file)
        
        p_sdpa_fw = float(p_rows[0][1])
        p_flex_fw = float(p_rows[1][1])
        r_sdpa_fw = float(r_rows[0][1])
        r_flex_fw = float(r_rows[1][1])
        
        p_sdpa_bw = float(p_rows[0][3])
        p_flex_bw = float(p_rows[1][3])
        r_sdpa_bw = float(r_rows[0][3])
        r_flex_bw = float(r_rows[1][3])
        
        comp_rows = [
            ["F.sdpa Forward", f"{p_sdpa_fw:.4f}", f"{r_sdpa_fw:.4f}", f"{r_sdpa_fw/p_sdpa_fw:.2f}x"],
            ["FlexAttn (indexing) Forward", f"{p_flex_fw:.4f}", f"{r_flex_fw:.4f}", f"{r_flex_fw/p_flex_fw:.2f}x"],
            ["F.sdpa Backward", f"{p_sdpa_bw:.4f}", f"{r_sdpa_bw:.4f}", f"{r_sdpa_bw/p_sdpa_bw:.2f}x"],
            ["FlexAttn (indexing) Backward", f"{p_flex_bw:.4f}", f"{r_flex_bw:.4f}", f"{r_flex_bw/p_flex_bw:.2f}x"],
        ]
        
        # Save comparison ratios to dictionary
        all_results["comparison"] = {
            "sdpa_fw_ratio": float(f"{r_sdpa_fw/p_sdpa_fw:.4f}"),
            "flex_fw_ratio": float(f"{r_flex_fw/p_flex_fw:.4f}"),
            "sdpa_bw_ratio": float(f"{r_sdpa_bw/p_sdpa_bw:.4f}"),
            "flex_bw_ratio": float(f"{r_flex_bw/p_flex_bw:.4f}"),
            "flops_ratio": float(f"{r_full_flops/p_full_flops:.4f}"),
        }
        
        # Print comparison table
        comp_table = tabulate(
            comp_rows,
            headers=[
                "Operation",
                "Padding Time (ms)",
                "One Batch Time (ms)",
                "One Batch/Padding Ratio",
            ],
            tablefmt="grid",
        )
        print(comp_table)
        
        # Write to markdown
        md_file.write(tabulate(
            comp_rows,
            headers=[
                "Operation",
                "Padding Time (ms)",
                "One Batch Time (ms)",
                "One Batch/Padding Ratio",
            ],
            tablefmt="pipe",
        ) + "\n\n")
        
        # Print analysis
        analysis_text = [
            f"- RIN with Padding uses {n_images} batch items with max length {p_max_len}",
            f"- RIN with One Batch uses 1 batch item with {r_Q} query tokens and {r_KV} KV tokens",
            f"- One Batch/Padding FLOPS ratio: {r_full_flops/p_full_flops:.2f}x"
        ]
        
        print("\nAnalysis:")
        for line in analysis_text:
            print(line)
        
        # Write to markdown
        md_file.write("### Analysis:\n\n")
        for line in analysis_text:
            md_file.write(f"{line}\n")
        
        # ======== EXPORT TO CSV ========
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write configuration
            writer.writerow(['Configuration'])
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Number of images', n_images])
            writer.writerow(['Latent tokens per image', latent_len])
            writer.writerow(['Min tokens', min_tokens])
            writer.writerow(['Max tokens', max_tokens])
            writer.writerow(['Heads', H])
            writer.writerow(['Dimension', D])
            writer.writerow(['Causal', causal])
            writer.writerow([])
            
            # Write data summary
            writer.writerow(['Data Summary'])
            for row in data_summary:
                writer.writerow(row)
            writer.writerow([])
            
            # Write padded results
            writer.writerow(['RIN with Padding Results'])
            writer.writerow([
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ])
            for row in p_rows:
                writer.writerow(row)
            writer.writerow([])
            
            # Write RIN results
            writer.writerow(['RIN with One Batch Results'])
            writer.writerow([
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ])
            for row in r_rows:
                writer.writerow(row)
            writer.writerow([])
            
            # Write comparison
            writer.writerow(['Comparison'])
            writer.writerow([
                "Operation",
                "Padding Time (ms)",
                "One Batch Time (ms)",
                "One Batch/Padding Ratio",
            ])
            for row in comp_rows:
                writer.writerow(row)
        
        # ======== EXPORT TO JSON ========
        with open(json_file, 'w') as jsonfile:
            json.dump(all_results, jsonfile, indent=2)
        
        # ======== CREATE VISUALIZATION ========
        # Create a bar chart comparing the timings
        labels = ['F.sdpa FW', 'FlexAttn FW', 'F.sdpa BW', 'FlexAttn BW']
        padded_times = [p_sdpa_fw, p_flex_fw, p_sdpa_bw, p_flex_bw]
        rin_times = [r_sdpa_fw, r_flex_fw, r_sdpa_bw, r_flex_bw]
        
        x = range(len(labels))
        width = 0.35
        
        fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
        
        ax.bar([i - width/2 for i in x], padded_times, width, label='RIN with Padding')
        ax.bar([i + width/2 for i in x], rin_times, width, label='RIN with One Batch')
        
        ax.set_ylabel('Time (ms)')
        ax.set_title('RIN with Padding vs One Batch Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add numerical values above bars
        for i, v in enumerate(padded_times):
            ax.text(i - width/2, v + 0.1, f"{v:.2f}", ha='center')
        
        for i, v in enumerate(rin_times):
            ax.text(i + width/2, v + 0.1, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(plot_file)
        
        # Add the visualization to the markdown
        md_file.write(f"\n\n### Performance Visualization\n\n")
        md_file.write(f"![Performance Comparison]({plot_file.name})\n\n")
        
        print(f"\nResults exported to:")
        print(f"- Markdown: {results_file}")
        print(f"- CSV: {csv_file}")
        print(f"- JSON: {json_file}")
        print(f"- Plot: {plot_file}")


def main(
    n_images: int = 128,
    latent_len: int = 128,
    min_tokens: int = 32,
    max_tokens: int = 256,
    heads: int = 16,
    dim: int = 64,
    causal: bool = False,
    check_correctness: bool = True,
    export_dir: str = "results",
):
    """
    Run combined benchmark with configurable parameters.
    """
    run_combined_benchmark(
        n_images=n_images,
        latent_len=latent_len,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        H=heads,
        D=dim,
        causal=causal,
        check_correctness=check_correctness,
        export_dir=export_dir,
    )


if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser, CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    
    CLI(main) 