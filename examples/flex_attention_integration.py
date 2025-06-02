"""
Simple example showing how to replace standard MHA with flex attention using one batch approach.
This demonstrates converting from batched sequences to concatenated sequences for flex attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from typing import List, Tuple

# Configure device and compilation
torch.set_default_device("cuda")
torch.manual_seed(42)

# Compile flex attention for better performance
flex_attention = torch.compile(flex_attention, dynamic=False)


class StandardMHA(nn.Module):
    """Standard Multi-Head Attention implementation using torch.nn.MultiheadAttention"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model] 
            value: [batch_size, seq_len_v, d_model]
            key_padding_mask: [batch_size, seq_len_k] - True for padding positions
        """
        output, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        return output


class FlexAttentionMHA(nn.Module):
    """Flex Attention MHA using one batch approach"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                seq_lengths_q: List[int], seq_lengths_kv: List[int]) -> Tuple[torch.Tensor, List[int]]:
        """
        One batch approach: concatenate all sequences into single batch
        
        Args:
            query: [total_q_tokens, d_model] - all query sequences concatenated
            key: [total_kv_tokens, d_model] - all key sequences concatenated  
            value: [total_kv_tokens, d_model] - all value sequences concatenated
            seq_lengths_q: List of query sequence lengths for each batch item
            seq_lengths_kv: List of key/value sequence lengths for each batch item
            
        Returns:
            output: [total_q_tokens, d_model] - concatenated output
            seq_lengths_q: sequence lengths for splitting output back into batches
        """
        total_q, d_model = query.shape
        total_kv = key.shape[0]
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # [total_q, d_model]
        K = self.k_proj(key)    # [total_kv, d_model] 
        V = self.v_proj(value)  # [total_kv, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(1, self.n_heads, total_q, self.d_head)   # [1, n_heads, total_q, d_head]
        K = K.view(1, self.n_heads, total_kv, self.d_head)  # [1, n_heads, total_kv, d_head]
        V = V.view(1, self.n_heads, total_kv, self.d_head)  # [1, n_heads, total_kv, d_head]
        
        # Create block mask for flex attention
        block_mask = self._create_block_mask(seq_lengths_q, seq_lengths_kv, total_q, total_kv)
        
        # Apply flex attention
        attn_output = flex_attention(Q, K, V, block_mask=block_mask)  # [1, n_heads, total_q, d_head]
        
        # Reshape and project output
        attn_output = attn_output.view(total_q, d_model)  # [total_q, d_model]
        output = self.out_proj(attn_output)
        
        return output, seq_lengths_q
    
    def _create_block_mask(self, seq_lengths_q: List[int], seq_lengths_kv: List[int], 
                          total_q: int, total_kv: int) -> torch.Tensor:
        """Create block mask that allows attention only within corresponding sequence pairs"""
        
        # Precompute sequence IDs for efficiency
        device = torch.cuda.current_device()
        
        # Create query sequence IDs
        q_seq_ids = torch.zeros(total_q, dtype=torch.long, device=device)
        q_offset = 0
        for i, length in enumerate(seq_lengths_q):
            q_seq_ids[q_offset:q_offset + length] = i
            q_offset += length
        
        # Create key/value sequence IDs  
        kv_seq_ids = torch.zeros(total_kv, dtype=torch.long, device=device)
        kv_offset = 0
        for i, length in enumerate(seq_lengths_kv):
            kv_seq_ids[kv_offset:kv_offset + length] = i
            kv_offset += length
        
        def mask_mod(b, h, q_idx, kv_idx):
            # Allow attention only within the same sequence
            kv_idx = torch.clamp(kv_idx, 0, total_kv - 1)
            return q_seq_ids[q_idx] == kv_seq_ids[kv_idx]
        
        return create_block_mask(mask_mod, 1, self.n_heads, total_q, total_kv, device=device)


def convert_batched_to_one_batch(batched_tensor: torch.Tensor, seq_lengths: List[int]) -> torch.Tensor:
    """Convert batched sequences to concatenated single batch"""
    # Remove padding and concatenate
    sequences = []
    for i, length in enumerate(seq_lengths):
        sequences.append(batched_tensor[i, :length])  # Remove padding
    return torch.cat(sequences, dim=0)


def convert_one_batch_to_batched(concatenated: torch.Tensor, seq_lengths: List[int]) -> torch.Tensor:
    """Convert concatenated sequences back to batched format with padding"""
    batch_size = len(seq_lengths)
    max_len = max(seq_lengths)
    d_model = concatenated.shape[1]
    
    # Create padded batch tensor
    batched = torch.zeros(batch_size, max_len, d_model, device=concatenated.device, dtype=concatenated.dtype)
    
    offset = 0
    for i, length in enumerate(seq_lengths):
        batched[i, :length] = concatenated[offset:offset + length]
        offset += length
    
    return batched


def demo_mha_replacement():
    """Demonstrate replacing standard MHA with flex attention"""
    
    # Configuration
    batch_size = 4
    d_model = 256
    n_heads = 8
    
    # Generate random sequence lengths
    seq_lengths_q = [32, 24, 40, 28]  # Query lengths
    seq_lengths_kv = [48, 36, 52, 44]  # Key/Value lengths
    max_len_q = max(seq_lengths_q)
    max_len_kv = max(seq_lengths_kv)
    
    # Create batched input data (with padding)
    query_batched = torch.randn(batch_size, max_len_q, d_model)
    key_batched = torch.randn(batch_size, max_len_kv, d_model)
    value_batched = torch.randn(batch_size, max_len_kv, d_model)
    
    # Create padding masks for standard MHA
    key_padding_mask = torch.zeros(batch_size, max_len_kv, dtype=torch.bool)
    for i, length in enumerate(seq_lengths_kv):
        key_padding_mask[i, length:] = True  # True for padding positions
    
    print("=== MHA Replacement Demo ===")
    print(f"Batch size: {batch_size}")
    print(f"Query lengths: {seq_lengths_q}")
    print(f"Key/Value lengths: {seq_lengths_kv}")
    print(f"Model dimension: {d_model}, Heads: {n_heads}")
    
    # Initialize both models
    standard_mha = StandardMHA(d_model, n_heads)
    flex_mha = FlexAttentionMHA(d_model, n_heads)
    
    print("\n--- Standard MHA ---")
    with torch.no_grad():
        standard_output = standard_mha(query_batched, key_batched, value_batched, key_padding_mask)
    print(f"Standard MHA output shape: {standard_output.shape}")
    
    print("\n--- Flex Attention MHA (One Batch) ---")
    # Convert to one batch format
    query_concat = convert_batched_to_one_batch(query_batched, seq_lengths_q)
    key_concat = convert_batched_to_one_batch(key_batched, seq_lengths_kv)
    value_concat = convert_batched_to_one_batch(value_batched, seq_lengths_kv)
    
    print(f"Concatenated query shape: {query_concat.shape}")
    print(f"Concatenated key shape: {key_concat.shape}")
    print(f"Concatenated value shape: {value_concat.shape}")
    
    with torch.no_grad():
        flex_output_concat, output_seq_lengths = flex_mha(
            query_concat, key_concat, value_concat, seq_lengths_q, seq_lengths_kv
        )
    print(f"Flex MHA concatenated output shape: {flex_output_concat.shape}")
    
    # Convert back to batched format for comparison
    flex_output_batched = convert_one_batch_to_batched(flex_output_concat, output_seq_lengths)
    print(f"Flex MHA batched output shape: {flex_output_batched.shape}")
    
    print("\n--- Comparison ---")
    print("Both approaches produce outputs with the same shape!")
    print("The flex attention approach is more memory efficient for variable-length sequences.")
    
    # Show memory usage comparison
    total_q_tokens = sum(seq_lengths_q)
    total_kv_tokens = sum(seq_lengths_kv)
    padded_q_tokens = batch_size * max_len_q
    padded_kv_tokens = batch_size * max_len_kv
    
    print(f"\nMemory efficiency:")
    print(f"Standard (padded): {padded_q_tokens} + {padded_kv_tokens} = {padded_q_tokens + padded_kv_tokens} tokens")
    print(f"Flex (concat): {total_q_tokens} + {total_kv_tokens} = {total_q_tokens + total_kv_tokens} tokens")
    print(f"Memory savings: {(1 - (total_q_tokens + total_kv_tokens) / (padded_q_tokens + padded_kv_tokens)):.1%}")


if __name__ == "__main__":
    demo_mha_replacement() 