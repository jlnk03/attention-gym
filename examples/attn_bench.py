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

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from triton.testing import do_bench

from attn_gym.masks.document_mask import length_to_offsets
from attn_gym.masks import (
    causal_mask,
    generate_sliding_window,
    generate_prefix_lm_mask,
    generate_doc_mask_mod,
)
from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap

import random
from itertools import accumulate


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
    "rin": lambda: run_rin_benchmark(),
}


torch.set_default_device("cuda")
torch.manual_seed(0)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

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
    print_mask: bool = True,
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

    causal_fa2 = lambda: F.scaled_dot_product_attention(*qkv, is_causal=True)
    sdpa_mask = lambda: F.scaled_dot_product_attention(*qkv, attn_mask=mask)
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
    test_mask(mask_mod=document_causal_mask, S=32768)


def _generate_rin_mask_mod(
    img_lengths: List[int], latent_len: int, device: str = "cuda"
) -> _mask_mod_signature:
    """
    Each image i owns:
      • latent_len query tokens              (Q side)
      • img_lengths[i] key/value tokens      (KV side)
    Queries may only see KV tokens from the same image.
    """
    # cumulative boundaries of the tape:  [0, L0, L0+L1, ...]
    boundaries = torch.tensor([0, *accumulate(img_lengths)], device=device, dtype=torch.long)
    total_kv   = int(boundaries[-1].item())

    def mask_mod(b, h, q_idx, kv_idx):
        # image id for latent query (fixed size per image)
        img_q  = q_idx // latent_len
        # image id for kv token (variable size -> searchsorted)
        kv_idx = torch.clamp(kv_idx, 0, total_kv - 1)
        img_kv = torch.searchsorted(boundaries, kv_idx, right=True) - 1
        return img_q == img_kv

    return mask_mod


def run_rin_benchmark(
    n_images: int = 4,
    latent_len: int = 128,
    min_tokens: int = 64,
    max_tokens: int = 512,
    B: int = 1,
    H: int = 16,
    D: int = 64,
    device: str = "cuda",
):
    """
    One latent block (latent_len) per image, variable-length image tapes.
    Benchmarks   full-SDPA   vs  SDPA+mask   vs  FlexAttention.
    """
    random.seed(0)
    img_lengths = [random.randint(min_tokens, max_tokens) for _ in range(n_images)]

    Q  = n_images * latent_len
    KV = sum(img_lengths)

    mask_mod   = _generate_rin_mask_mod(img_lengths, latent_len, device)

    print(f"Q={Q}, KV={KV}")
    print(f"img_lengths={img_lengths}")
    print(mask_mod)
    # print(f'mask mod shape: {mask_mod.shape}')
    block_mask = create_block_mask_cached(mask_mod, 1, 1, Q, KV, device=device)
    print(f"block_mask={block_mask}")

    test_mask(mask_mod=mask_mod, S=KV)
    # dense_mask = create_mask(mask_mod, 1, 1, Q, KV, device=device)

    # q = torch.randn(B, H, Q,  D, device=device, dtype=data_type, requires_grad=True)
    # k = torch.randn(B, H, KV, D, device=device, dtype=data_type, requires_grad=True)
    # v = torch.randn_like(k)
    # dO= torch.randn_like(q)

    # # full   = lambda: F.scaled_dot_product_attention(q, k, v)
    # # masked = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask)
    # flex   = lambda: flex_attention(q, k, v, block_mask=block_mask)

    # density = (100 - block_mask.sparsity()) / 100
    # full_flops   = B * H * D * Q * KV
    # sparse_flops = density * full_flops

    # rows = []
    # for fn, label, flop in ((flex,   "flexattention", sparse_flops)):
    #     fw = do_bench(fn)
    #     out = fn()
    #     bw = do_bench(lambda: out.backward(dO, retain_graph=True))
    #     rows.append([label, f"{fw:.3f}", f"{calculate_tflops(flop, fw, 4):.2f}",
    #                          f"{bw:.3f}", f"{calculate_tflops(flop, bw, 10):.2f}"])
    #     out.detach_(); q.grad=None; k.grad=None; v.grad=None; torch.cuda.empty_cache()

    # label = "flexattention"
    # flop = sparse_flops
    # fw = do_bench(flex)
    # out = flex()
    # bw = do_bench(lambda: out.backward(dO, retain_graph=True))
    # rows.append([label, f"{fw:.3f}", f"{calculate_tflops(flop, fw, 4):.2f}",
    #                         f"{bw:.3f}", f"{calculate_tflops(flop, bw, 10):.2f}"])
    # out.detach_(); q.grad=None; k.grad=None; v.grad=None; torch.cuda.empty_cache()


    # print_header("RIN Cross-Attention (B=1)")
    # print(tabulate(rows, headers=["Op","FW ms","FW TF/s","BW ms","BW TF/s"], tablefmt="grid"))
    # print(f"\nBlockMask sparsity: {block_mask.sparsity():.2f}%   (Q={Q}, KV={KV})")

    # quick correctness (masked vs flex)
    # torch.testing.assert_close(masked(), flex(), atol=1e-1, rtol=1e-2)
    # print("Correctness check passed ✅")


def main(examples: List[str] = ["all"]):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """

    if "all" in examples:
        ex_to_run = list(AVAILABLE_EXAMPLES.keys())
    else:
        ex_to_run = examples

    for ex in ex_to_run:
        if ex in AVAILABLE_EXAMPLES:
            AVAILABLE_EXAMPLES[ex]()
            torch.cuda.empty_cache()
        else:
            print(f"Warning: Unknown example key '{ex}'. Skipping.")


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
