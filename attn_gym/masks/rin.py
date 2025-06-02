"""Mask Mod for Retrieval-in-Network (RIN) Cross-Attention"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature
from typing import List
from itertools import accumulate


def generate_rin_mask_mod(
    img_lengths: List[int], latent_len: int, device: str = "cuda"
) -> _mask_mod_signature:
    """
    Generates a mask mod for RIN Cross-Attention.

    Each image i owns:
      • latent_len query tokens (Q side)
      • img_lengths[i] key/value tokens (KV side)
    Queries may only see KV tokens from the same image.

    Args:
        img_lengths: List of token lengths for each image.
        latent_len: Number of latent tokens per image.
        device: Device to place tensors on.
    """
    # cumulative boundaries of the tape: [0, L0, L0+L1, ...]
    boundaries = torch.tensor([0, *accumulate(img_lengths)], device=device, dtype=torch.long)
    total_kv = int(boundaries[-1].item())

    def rin_mask_mod(b, h, q_idx, kv_idx):
        # image id for latent query (fixed size per image)
        img_q = q_idx // latent_len
        # image id for kv token (variable size -> searchsorted)
        kv_idx = torch.clamp(kv_idx, 0, total_kv - 1)
        img_kv = torch.searchsorted(boundaries, kv_idx, right=True) - 1
        return img_q == img_kv

    return rin_mask_mod


def main(device: str = "cpu"):
    """
    Demonstrate the usage of the RIN Cross-Attention mask mod.

    In this case we would generate a mask of latent queries × sum(img_lengths)
    assuming 3 images with latent_len=2 and variable image token lengths [3, 2, 4]

            img1       img2     img3
        L1  █ █ █ |   ░ ░   |  ░ ░ ░ ░  
        L1  █ █ █ |   ░ ░   |  ░ ░ ░ ░  
        L2  ░ ░ ░ |   █ █   |  ░ ░ ░ ░  
        L2  ░ ░ ░ |   █ █   |  ░ ░ ░ ░  
        L3  ░ ░ ░ |   ░ ░   |  █ █ █ █  
        L3  ░ ░ ░ |   ░ ░   |  █ █ █ █  
    """
    from attn_gym import visualize_attention_scores

    latent_len = 128  # Number of latent tokens per image
    img_lengths = [64, 128, 72]  # Token lengths for each image
    n_images = len(img_lengths)
    
    total_latent_tokens = n_images * latent_len  # Total number of query tokens
    total_image_tokens = sum(img_lengths)  # Total number of key/value tokens

    B, H, HEAD_DIM = 1, 1, 8

    def make_tensor(seq_len):
        return torch.ones(B, H, seq_len, HEAD_DIM, device=device)

    query = make_tensor(total_latent_tokens)
    key = make_tensor(total_image_tokens)

    rin_mask = generate_rin_mask_mod(img_lengths, latent_len, device)

    visualize_attention_scores(
        query,
        key,
        mask_mod=rin_mask,
        device=device,
        name="rin_cross_attention_mask",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .[viz]")
    CLI(main) 