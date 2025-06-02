from functools import lru_cache
from typing import Optional, List

import torch
import torch.nn.functional as F

from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
    _score_mod_signature,
    _mask_mod_signature,
)

from triton.testing import do_bench

from attn_gym.masks.document_mask import length_to_offsets
from attn_gym.masks import (
    causal_mask,
    generate_sliding_window,
    generate_prefix_lm_mask,
    generate_doc_mask_mod,
)
from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap

import json
import datetime
from pathlib import Path
import numpy as np


AVAILABLE_EXAMPLES = {
    "causal": lambda: test_mask(mask_mod=causal_mask),
    "alibi": lambda: test_mask(score_mod=generate_alibi_bias(16), skip_correctness=True),
    "sliding_window": lambda: test_mask(mask_mod=generate_sliding_window(window_size=1024)),
    "prefix_lm": lambda: test_mask(mask_mod=generate_prefix_lm_mask(prefix_length=1024)),
    "document": lambda: run_document_masking(max_seq_len=32768, num_docs=12),
    "softcap": lambda: test_mask(
        score_mod=generate_tanh_softcap(30, approx=False), skip_correctness=True
    ),
    "softcap_approx": lambda: test_mask(
        score_mod=generate_tanh_softcap(30, approx=True), skip_correctness=True
    ),
    "naked": lambda: test_naked_attention(),
    "naked_cross": lambda: test_naked_cross_attention(),
}


torch.set_default_device("cuda")
torch.manual_seed(0)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)
sdpa = torch.compile(F.scaled_dot_product_attention, dynamic=False)

# For better performance, you can use:
# flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

data_type = torch.float16

# The kernels will utilize block sparsity to increase performance
print(f"Using the default sparsity block size: {_DEFAULT_SPARSE_BLOCK_SIZE}")


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12


def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")


def test_mask(
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    B: int = 16,
    H: int = 16,
    S: int = 8192,
    D: int = 64,
    skip_correctness: bool = False,
    print_mask: bool = False,
    device: str = "cuda",
):
    assert score_mod is not None or mask_mod is not None, "Must provide a score_mod or mask_mod"
    if mask_mod is not None:
        block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=device)
    else:
        block_mask = None
    sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
    mask = create_mask(sdpa_mask_fn, 1, 1, S, S, device=device)

    qkv = [
        torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True)
        for _ in range(3)
    ]
    gradOut = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    causal_fa2 = lambda: sdpa(*qkv, is_causal=True)
    sdpa_mask = lambda: sdpa(*qkv, attn_mask=mask)
    flex_attention_call = lambda: flex_attention(*qkv, score_mod=score_mod, block_mask=block_mask)

    results = []
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0
    causal_fav2_flops = 0.5 * B * H * D * S * S
    flops = density * B * H * D * S * S

    times = []
    for attn in (causal_fa2, sdpa_mask, flex_attention_call):
        fwd_time = do_bench(attn)
        fwd_out = attn()
        bwd_time = do_bench(lambda: fwd_out.backward(gradOut, retain_graph=True))  # noqa: F821
        times.append((fwd_time, bwd_time))

        del fwd_out
        torch.cuda.empty_cache()

    print_header(
        f"{score_mod.__name__ if score_mod is not None else mask_mod.__name__}".replace(
            "_", " "
        ).title()
    )
    # Inline correctness check
    if not skip_correctness:
        sdpa_mask_outs = []
        flex_outs = []

        for tensor in qkv:
            tensor.grad = None

        out1 = sdpa_mask()
        sdpa_mask_outs.append(out1)
        out1.backward(gradOut)
        sdpa_mask_outs += [tensor.grad for tensor in qkv]

        for tensor in qkv:
            tensor.grad = None

        out2 = flex_attention_call()
        flex_outs.append(out2)
        out2.backward(gradOut)
        flex_outs += [tensor.grad for tensor in qkv]
        for flex, sdpa_mask in zip(flex_outs, sdpa_mask_outs):
            torch.testing.assert_close(flex, sdpa_mask, atol=1e-1, rtol=1e-2)

        print("Correctness check passed ✅")

    (
        (causal_fa2_time, causal_fa2_bw_time),
        (sdpa_mask_time, sdpa_mask_bw_time),
        (flex_ms, flex_bw_ms),
    ) = times
    # Usage in your results formatting:
    results = [
        [
            "causal FA2",
            f"{causal_fa2_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_time, 4):.2f}",
            f"{causal_fa2_bw_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10):.2f}",
        ],
        [
            "F.sdpa + mask",
            f"{sdpa_mask_time:.4f}",
            f"{calculate_tflops(flops, sdpa_mask_time, 4):.2f}",
            f"{sdpa_mask_bw_time:.4f}",
            f"{calculate_tflops(flops, sdpa_mask_bw_time, 10):.2f}",
        ],
        [
            "flexattention",
            f"{flex_ms:.4f}",
            f"{calculate_tflops(flops, flex_ms, 4):.2f}",
            f"{flex_bw_ms:.4f}",
            f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
        ],
    ]
    print(
        tabulate(
            results,
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
    if print_mask:
        print(f"\nBlock Mask:\n{block_mask}")

    # Prepare detailed results for plotting
    detailed_results_for_plot = [
        {
            "operation": "causal FA2",
            "fw_time": causal_fa2_time,
            "fw_tflops": calculate_tflops(causal_fav2_flops, causal_fa2_time, 4),
            "bw_time": causal_fa2_bw_time,
            "bw_tflops": calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10),
        },
        {
            "operation": "F.sdpa + mask",
            "fw_time": sdpa_mask_time,
            "fw_tflops": calculate_tflops(flops, sdpa_mask_time, 4),
            "bw_time": sdpa_mask_bw_time,
            "bw_tflops": calculate_tflops(flops, sdpa_mask_bw_time, 10),
        },
        {
            "operation": "flexattention",
            "fw_time": flex_ms,
            "fw_tflops": calculate_tflops(flops, flex_ms, 4),
            "bw_time": flex_bw_ms,
            "bw_tflops": calculate_tflops(flops, flex_bw_ms, 10),
        },
    ]
    return detailed_results_for_plot


def run_document_masking(max_seq_len: int, num_docs: int):
    import random

    random.seed(0)

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    lengths = generate_random_lengths(max_seq_len, num_docs)
    offsets = length_to_offsets(lengths, "cuda")
    document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)
    return test_mask(mask_mod=document_causal_mask, S=max_seq_len)


def save_to_unified_json(all_benchmark_data, json_file="benchmark_results.json"):
    """Save benchmark results to a unified JSON file, updating existing data with new results."""
    json_path = Path(json_file)
    
    # Load existing data if file exists
    existing_data = {}
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = {}
    
    # Create run metadata
    timestamp = datetime.datetime.now().isoformat()
    run_id = f"run_{timestamp.replace(':', '-').replace('.', '-')}"
    
    # Add new benchmark run data
    for item in all_benchmark_data:
        example_name = item["example_name"]
        data = item["data"]
        
        if not data:
            print(f"No data for example: {example_name}. Skipping.")
            continue
        
        # Initialize example if it doesn't exist
        if example_name not in existing_data:
            existing_data[example_name] = {
                "description": f"Benchmark results for {example_name}",
                "runs": {}
            }
        
        # Add this run's data
        existing_data[example_name]["runs"][run_id] = {
            "timestamp": timestamp,
            "results": data,
            "metadata": {
                "device": "cuda",
                "data_type": str(data_type),
                "torch_version": torch.__version__,
                "num_operations": len(data)
            }
        }
        
        # Keep only the latest run as "latest" for easy access
        existing_data[example_name]["latest"] = existing_data[example_name]["runs"][run_id]
    
    # Save updated data
    with open(json_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"Benchmark results saved to: {json_path.resolve()}")
    print(f"Run ID: {run_id}")
    return str(json_path.resolve())


def test_naked_attention(
    B: int = 16,
    H: int = 16,
    S: int = 16384,
    D: int = 64,
    skip_correctness: bool = False,
    device: str = "cuda",
):


    qkv = [
        torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True)
        for _ in range(3)
    ]
    gradOut = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    causal_fa2 = lambda: sdpa(*qkv, is_causal=True)
    sdpa_mask = lambda: sdpa(*qkv)
    flex_attention_call = lambda: flex_attention(*qkv)

    results = []

    density = 1.0
    causal_fav2_flops = 0.5 * B * H * D * S * S
    flops = density * B * H * D * S * S

    times = []
    for attn in (causal_fa2, sdpa_mask, flex_attention_call):
        fwd_time = do_bench(attn)
        fwd_out = attn()
        bwd_time = do_bench(lambda: fwd_out.backward(gradOut, retain_graph=True))  # noqa: F821
        times.append((fwd_time, bwd_time))

        del fwd_out
        torch.cuda.empty_cache()

    print_header("Naked Self Attention (No Masking)")
    # Inline correctness check
    if not skip_correctness:
        sdpa_mask_outs = []
        flex_outs = []

        for tensor in qkv:
            tensor.grad = None

        out1 = sdpa_mask()
        sdpa_mask_outs.append(out1)
        out1.backward(gradOut)
        sdpa_mask_outs += [tensor.grad for tensor in qkv]

        for tensor in qkv:
            tensor.grad = None

        out2 = flex_attention_call()
        flex_outs.append(out2)
        out2.backward(gradOut)
        flex_outs += [tensor.grad for tensor in qkv]
        for flex, sdpa_mask in zip(flex_outs, sdpa_mask_outs):
            torch.testing.assert_close(flex, sdpa_mask, atol=1e-1, rtol=1e-2)

        print("Correctness check passed ✅")

    (
        (causal_fa2_time, causal_fa2_bw_time),
        (sdpa_mask_time, sdpa_mask_bw_time),
        (flex_ms, flex_bw_ms),
    ) = times
    # Usage in your results formatting:
    results = [
        [
            "causal FA2",
            f"{causal_fa2_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_time, 4):.2f}",
            f"{causal_fa2_bw_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10):.2f}",
        ],
        [
            "F.sdpa + mask",
            f"{sdpa_mask_time:.4f}",
            f"{calculate_tflops(flops, sdpa_mask_time, 4):.2f}",
            f"{sdpa_mask_bw_time:.4f}",
            f"{calculate_tflops(flops, sdpa_mask_bw_time, 10):.2f}",
        ],
        [
            "flexattention",
            f"{flex_ms:.4f}",
            f"{calculate_tflops(flops, flex_ms, 4):.2f}",
            f"{flex_bw_ms:.4f}",
            f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
        ],
    ]
    print(
        tabulate(
            results,
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

    # Prepare detailed results for plotting
    detailed_results_for_plot = [
        {
            "operation": "causal FA2",
            "fw_time": causal_fa2_time,
            "fw_tflops": calculate_tflops(causal_fav2_flops, causal_fa2_time, 4),
            "bw_time": causal_fa2_bw_time,
            "bw_tflops": calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10),
        },
        {
            "operation": "F.sdpa + mask",
            "fw_time": sdpa_mask_time,
            "fw_tflops": calculate_tflops(flops, sdpa_mask_time, 4),
            "bw_time": sdpa_mask_bw_time,
            "bw_tflops": calculate_tflops(flops, sdpa_mask_bw_time, 10),
        },
        {
            "operation": "flexattention",
            "fw_time": flex_ms,
            "fw_tflops": calculate_tflops(flops, flex_ms, 4),
            "bw_time": flex_bw_ms,
            "bw_tflops": calculate_tflops(flops, flex_bw_ms, 10),
        },
    ]
    return detailed_results_for_plot


def test_naked_cross_attention(
    B: int = 16,
    H: int = 16,
    S_q: int = 2048,  # Query sequence length
    S_kv: int = 8192,  # Key/Value sequence length
    D: int = 64,
    skip_correctness: bool = False,
    device: str = "cuda",
):
    """Test naked cross attention where query and key/value come from different sequences."""
    
    # Create separate tensors for query and key/value
    q = torch.randn(B, H, S_q, D, device=device, dtype=data_type, requires_grad=True)
    k = torch.randn(B, H, S_kv, D, device=device, dtype=data_type, requires_grad=True)
    v = torch.randn(B, H, S_kv, D, device=device, dtype=data_type, requires_grad=True)
    
    gradOut = torch.randn(B, H, S_q, D, device=device, dtype=torch.float16)

    # For cross attention, we don't use is_causal since query and key/value are different sequences
    sdpa_cross = lambda: sdpa(q, k, v)
    flex_attention_cross = lambda: flex_attention(q, k, v)

    density = 1.0
    # FLOPS calculation for cross attention: B * H * D * S_q * S_kv
    flops = density * B * H * D * S_q * S_kv

    times = []
    for attn in (sdpa_cross, flex_attention_cross):
        fwd_time = do_bench(attn)
        fwd_out = attn()
        bwd_time = do_bench(lambda: fwd_out.backward(gradOut, retain_graph=True))
        times.append((fwd_time, bwd_time))

        del fwd_out
        torch.cuda.empty_cache()

    print_header("Naked Cross Attention (No Masking)")
    
    # Inline correctness check
    if not skip_correctness:
        sdpa_outs = []
        flex_outs = []

        # Clear gradients
        for tensor in [q, k, v]:
            tensor.grad = None

        out1 = sdpa_cross()
        sdpa_outs.append(out1)
        out1.backward(gradOut)
        sdpa_outs += [q.grad, k.grad, v.grad]

        # Clear gradients again
        for tensor in [q, k, v]:
            tensor.grad = None

        out2 = flex_attention_cross()
        flex_outs.append(out2)
        out2.backward(gradOut)
        flex_outs += [q.grad, k.grad, v.grad]
        
        for flex, sdpa_it in zip(flex_outs, sdpa_outs):
            torch.testing.assert_close(flex, sdpa_it, atol=1e-1, rtol=1e-2)

        print("Correctness check passed ✅")

    (
        (sdpa_time, sdpa_bw_time),
        (flex_ms, flex_bw_ms),
    ) = times
    
    # Format results for display
    results = [
        [
            "F.sdpa cross",
            f"{sdpa_time:.4f}",
            f"{calculate_tflops(flops, sdpa_time, 4):.2f}",
            f"{sdpa_bw_time:.4f}",
            f"{calculate_tflops(flops, sdpa_bw_time, 10):.2f}",
        ],
        [
            "flexattention cross",
            f"{flex_ms:.4f}",
            f"{calculate_tflops(flops, flex_ms, 4):.2f}",
            f"{flex_bw_ms:.4f}",
            f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
        ],
    ]
    print(
        tabulate(
            results,
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

    # Prepare detailed results for plotting
    detailed_results_for_plot = [
        {
            "operation": "F.sdpa cross",
            "fw_time": sdpa_time,
            "fw_tflops": calculate_tflops(flops, sdpa_time, 4),
            "bw_time": sdpa_bw_time,
            "bw_tflops": calculate_tflops(flops, sdpa_bw_time, 10),
        },
        {
            "operation": "flexattention cross",
            "fw_time": flex_ms,
            "fw_tflops": calculate_tflops(flops, flex_ms, 4),
            "bw_time": flex_bw_ms,
            "bw_tflops": calculate_tflops(flops, flex_bw_ms, 10),
        },
    ]
    return detailed_results_for_plot


def main(examples: List[str] = ["all"]):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """

    if "all" in examples:
        ex_to_run = list(AVAILABLE_EXAMPLES.keys())
    else:
        ex_to_run = examples

    all_benchmark_data = []
    for ex in ex_to_run:
        if ex in AVAILABLE_EXAMPLES:
            print(f"\nRunning example: {ex}")
            current_run_results = AVAILABLE_EXAMPLES[ex]()
            if current_run_results:
                all_benchmark_data.append({"example_name": ex, "data": current_run_results})
            torch.cuda.empty_cache()
        else:
            print(f"Warning: Unknown example key '{ex}'. Skipping.")

    if all_benchmark_data:
        save_to_unified_json(all_benchmark_data)
    else:
        print("No benchmark data collected to save.")


if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    parser = ArgumentParser(description="Run specific examples or all examples.")
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        default=["all"],
        help="List of examples to run. Use space to separate multiple examples. "
        "Available options: "
        + ", ".join(sorted(AVAILABLE_EXAMPLES.keys()))
        + ", or 'all' to run all examples.",
    )

    args = parser.parse_args()
    main(**vars(args))
