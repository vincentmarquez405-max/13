#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPLETE: K-L Prior Ensemble with Dreamer + RAG + Real Mamba
------------------------------------------------------------
Now includes ALL missing components:
  1. K-L Dreamer (DreamerV3-style world model pilot)
  2. K-L RAG (external knowledge retrieval)
  3. Real Mamba SSM (with GRU fallback)
  4. Full architecture from the diagram

Usage:
  python3 complete_solution.py --epochs 50 --use-rag --knowledge-size 100
"""

import math
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

# ------------------------------------
# Configuration Constants
# ------------------------------------

# RAG Configuration
RAG_INTEGRATION_WEIGHT = 0.1
RAG_TRIGGER_THRESHOLD = 0.5
RAG_TOP_K = 3

# Loss Weights
RECON_LOSS_WEIGHT = 0.1
REWARD_LOSS_WEIGHT = 0.05

# Memory Management
CACHE_REFRESH_INTERVAL = 25
GRADIENT_CLIP_NORM = 1.0

# Numerical Stability
MIN_TAU = 1e-12
MIN_NORM_EPSILON = 1e-12
EIGENVALUE_MIN = 0.0

# ------------------------------------
# K-L Math Components
# ------------------------------------

def positional_encoding(T: int, d: int, device: torch.device) -> torch.Tensor:
    """
    Generate sinusoidal positional encodings.

    Args:
        T: Sequence length
        d: Dimension of encoding
        device: Target device for tensor

    Returns:
        Positional encoding tensor of shape (T, d)
    """
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

def _time_kernel(t: torch.Tensor, tau: float, kind: str = "exp") -> torch.Tensor:
    """
    Compute time-based kernel matrix.

    Args:
        t: Time points tensor
        tau: Kernel bandwidth parameter
        kind: Kernel type ('exp' or 'gauss')

    Returns:
        Symmetric kernel matrix

    Raises:
        ValueError: If kernel type is invalid
    """
    dt = (t[:, None] - t[None, :]).abs()
    tau = max(float(tau), MIN_TAU)
    if kind == "exp":
        K = torch.exp(-dt / tau)
    elif kind == "gauss":
        K = torch.exp(-(dt * dt) / (2.0 * tau * tau))
    else:
        raise ValueError(f"kernel must be 'exp' or 'gauss', got '{kind}'")
    return 0.5 * (K + K.T)

def _compute_kl_modes(
    H_full: torch.Tensor,
    t_full: torch.Tensor,
    n_components: int,
    tau: float,
    kernel: str
) -> Optional[torch.Tensor]:
    """
    Compute K-L expansion modes from historical hidden states.

    Args:
        H_full: Historical hidden states of shape (T, d_model)
        t_full: Timestamps of shape (T,)
        n_components: Number of K-L components to extract
        tau: Time kernel bandwidth parameter
        kernel: Kernel type, either 'exp' or 'gauss'

    Returns:
        K-L modes of shape (n_components, d_model), or None if computation fails

    Raises:
        ValueError: If kernel is not 'exp' or 'gauss'
    """
    T = H_full.shape[0]
    if T < max(n_components + 4, 8):
        logger.debug(f"Insufficient history length {T} for {n_components} components")
        return None

    span = max(float(t_full[-1].item() - t_full[0].item()), 1e-6)
    t_norm = ((t_full - t_full[0]) / span).to(dtype=torch.float64)
    K = _time_kernel(t_norm, tau=(tau / max(T, 1.0)), kind=kernel).to(dtype=torch.float64)
    K = 0.5 * (K + K.T)
    K = K.contiguous()
    eps = 1e-8 if T <= 2048 else 1e-6
    K = K + eps * torch.eye(T, dtype=K.dtype, device=K.device)

    try:
        evals, evecs = torch.linalg.eigh(K)
    except (torch.linalg.LinAlgError, RuntimeError) as e:
        logger.warning(f"eigh failed: {e}, falling back to SVD")
        try:
            U, S, _ = torch.linalg.svd(K, full_matrices=False)
            evals, evecs = S, U
        except (torch.linalg.LinAlgError, RuntimeError) as e:
            logger.error(f"SVD also failed: {e}")
            return None

    idx = torch.argsort(evals, descending=True)[:n_components]
    lams = torch.clamp(evals[idx], min=EIGENVALUE_MIN)
    phi  = evecs[:, idx]
    phi  = phi / (phi.norm(dim=0, keepdim=True) + MIN_NORM_EPSILON)

    coeffs = (phi.detach().T @ H_full.to(phi.dtype))
    W = torch.sqrt(lams.detach() + MIN_NORM_EPSILON)[:, None]
    M = W * coeffs
    return M

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = positional_encoding(max_len, d_model, torch.device('cpu'))
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        T = x.shape[0]
        if offset + T > self.max_len:
            raise ValueError(
                f"Sequence length {offset + T} exceeds max_len {self.max_len}. "
                f"offset={offset}, T={T}"
            )
        return x + self.pe[offset:offset+T].unsqueeze(1).to(x.device)

# ------------------------------------
# NEW: K-L RAG Module
# ------------------------------------

class K_L_RAG(nn.Module):
    """
    External knowledge retrieval system.
    Uses simple embedding similarity for retrieval.
    """
    def __init__(self, d_model: int, n_tokens: int = 2, knowledge_size: int = 100):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.knowledge_size = knowledge_size
        
        # Simple knowledge base: learnable embeddings
        self.knowledge_embeddings = nn.Parameter(
            torch.randn(knowledge_size, d_model) * 0.02
        )
        self.knowledge_keys = nn.Parameter(
            torch.randn(knowledge_size, d_model) * 0.02
        )
        
        # Query projection
        self.query_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, n_tokens * d_model)
        
    def retrieve(self, query_state: torch.Tensor, top_k: int = None) -> torch.Tensor:
        """
        Retrieve relevant knowledge based on query state.

        Args:
            query_state: Query tensor of shape (B, d_model) or (d_model,)
            top_k: Number of top knowledge items to retrieve (default: RAG_TOP_K)

        Returns:
            Retrieved knowledge tokens of shape (n_tokens, B, d_model)
        """
        if top_k is None:
            top_k = RAG_TOP_K

        if query_state.dim() == 1:
            query_state = query_state.unsqueeze(0)  # (1, d_model)

        B = query_state.shape[0]
        device = query_state.device

        # Project query
        query = self.query_proj(query_state)  # (B, d_model)

        # Compute similarity with knowledge base
        # query: (B, d_model), keys: (K, d_model)
        scores = torch.matmul(query, self.knowledge_keys.T)  # (B, K)
        scores = scores / math.sqrt(self.d_model)

        # Get top-k
        top_k = min(top_k, self.knowledge_size)
        topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=-1)  # (B, top_k)

        # Soft retrieval (weighted average) - VECTORIZED
        weights = F.softmax(topk_scores, dim=-1)  # (B, top_k)

        # Gather knowledge embeddings - VECTORIZED using advanced indexing
        # topk_indices: (B, top_k) -> gather embeddings: (B, top_k, d_model)
        gathered_embeddings = self.knowledge_embeddings[topk_indices]  # (B, top_k, d_model)

        # Weighted sum: (B, top_k, 1) * (B, top_k, d_model) -> (B, d_model)
        retrieved = torch.sum(weights.unsqueeze(-1) * gathered_embeddings, dim=1)

        # Project to value
        value = self.value_proj(retrieved)  # (B, d_model)

        # Project to tokens
        tokens_flat = self.out_proj(value)  # (B, n_tokens * d_model)
        tokens = tokens_flat.reshape(B, self.n_tokens, self.d_model)  # (B, n_tokens, d_model)
        tokens = tokens.permute(1, 0, 2)  # (n_tokens, B, d_model)

        return tokens

# ------------------------------------
# NEW: K-L Dreamer (World Model Pilot)
# ------------------------------------

class K_L_Dreamer(nn.Module):
    """
    DreamerV3-inspired world model pilot.
    - Learns to predict next states
    - Decides when to trigger RAG
    - Provides policy for the lane
    """
    def __init__(self, d_model: int, d_state: int = 32, d_action: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_action = d_action
        
        # === World Model Components ===
        
        # 1. Encoder: observation → latent state
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_state * 2),
            nn.LayerNorm(d_state * 2),
            nn.ReLU(),
            nn.Linear(d_state * 2, d_state)
        )
        
        # 2. RSSM (Recurrent State Space Model)
        self.rssm = nn.GRUCell(d_state + d_action, d_state)
        
        # 3. Decoder: latent state → predicted observation
        self.decoder = nn.Sequential(
            nn.Linear(d_state, d_state * 2),
            nn.ReLU(),
            nn.Linear(d_state * 2, d_model)
        )
        
        # 4. Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(d_state, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # === Policy Components ===
        
        # 5. Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(d_state, 64),
            nn.ReLU(),
            nn.Linear(64, d_action)
        )
        
        # 6. Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(d_state, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # State buffer
        self.register_buffer("h_dream", torch.zeros(1, d_state))
        
    def reset_state(self, B: int, device):
        """Reset dream state."""
        self.h_dream = torch.zeros(B, self.d_state, device=device)
        
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state."""
        return self.encoder(obs)
        
    def imagine_step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state given action (world model)."""
        state_action = torch.cat([state, action], dim=-1)
        next_state = self.rssm(state_action, state)
        return next_state
        
    def decode(self, state: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observation."""
        return self.decoder(state)
        
    def compute_policy(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute policy actions.
        Returns dict with:
          - 'action': continuous action vector
          - 'trigger_rag': probability to trigger RAG
          - 'value': state value estimate
        """
        # Get policy distribution
        action_logits = self.policy_net(state)  # (B, d_action)
        action = torch.tanh(action_logits)  # Squash to [-1, 1]
        
        # First action dimension controls RAG triggering
        trigger_rag = torch.sigmoid(action_logits[:, 0])  # (B,)
        
        # Value estimate
        value = self.value_net(state).squeeze(-1)  # (B,)
        
        return {
            'action': action,
            'trigger_rag': trigger_rag,
            'value': value
        }
        
    def forward(self, obs: torch.Tensor, update_state: bool = True) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode → policy → imagine next state.
        obs: (B, d_model) - current observation from Mamba
        """
        B = obs.shape[0]
        device = obs.device
        
        if self.h_dream.shape[0] != B:
            self.reset_state(B, device)
        
        # 1. Encode current observation
        z = self.encode(obs)  # (B, d_state)
        
        # 2. Compute policy
        policy_out = self.compute_policy(z)
        
        # 3. Imagine next state (world model prediction)
        if update_state:
            next_state = self.imagine_step(z, policy_out['action'])
            self.h_dream = next_state.detach()
        
        # 4. Decode for reconstruction loss
        obs_recon = self.decode(z)
        
        return {
            **policy_out,
            'state': z,
            'obs_recon': obs_recon,
            'reward_pred': self.reward_head(z).squeeze(-1)
        }

# ------------------------------------
# NEW: Real Mamba Core (with fallback)
# ------------------------------------

class MambaCore(nn.Module):
    """
    Real Mamba SSM if available, else GRU fallback.
    """
    def __init__(self, d_model: int, n_layers: int = 2, use_real_mamba: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_real_mamba = use_real_mamba
        
        if use_real_mamba:
            try:
                from mamba_ssm import Mamba
                self.layers = nn.ModuleList([
                    Mamba(
                        d_model=d_model,
                        d_state=16,
                        d_conv=4,
                        expand=2
                    ) for _ in range(n_layers)
                ])
                self.is_mamba = True
                print("✅ Using Real Mamba SSM")
            except ImportError:
                print("⚠️  mamba-ssm not found, using GRU fallback")
                self.is_mamba = False
                self.gru = nn.GRU(d_model, d_model, n_layers, batch_first=False)
        else:
            self.is_mamba = False
            self.gru = nn.GRU(d_model, d_model, n_layers, batch_first=False)
            
        self.ln = nn.LayerNorm(d_model)
        self.h_state = None
        
    def forward(self, x_chunk, h_state=None):
        """
        x_chunk: (T, B, d_model)
        h_state: hidden state (for GRU) or None (for Mamba)
        """
        if self.is_mamba:
            # Mamba processes sequence
            x = x_chunk
            for layer in self.layers:
                # Mamba expects (B, T, D)
                x_bt = x.permute(1, 0, 2)  # (B, T, D)
                x_bt = layer(x_bt)
                x = x_bt.permute(1, 0, 2)  # (T, B, D)
            x_out = self.ln(x)
            h_next = None  # Mamba doesn't return explicit state
        else:
            # GRU fallback
            if h_state is None:
                h_state = torch.zeros(
                    self.gru.num_layers, 
                    x_chunk.shape[1], 
                    self.d_model,
                    device=x_chunk.device
                )
            x_out, h_next = self.gru(x_chunk, h_state)
            x_out = self.ln(x_out)
            
        return x_out, h_next

# ------------------------------------
# Global History Buffer (Same as before)
# ------------------------------------

class GlobalHistoryBuffer(nn.Module):
    """
    Circular buffer for global history management.
    Maintains a fixed-size window of historical hidden states and timestamps.
    """
    def __init__(self, depth: int, d_model: int):
        super().__init__()
        self.depth = depth
        self.d_model = d_model

        self.register_buffer("_times", torch.zeros(0, dtype=torch.float32))
        self.register_buffer("_hist",  torch.zeros(0, d_model, dtype=torch.float32))

    def append(self, x_chunk: torch.Tensor, offset_t: int) -> None:
        """
        Append new chunk to history buffer.

        Args:
            x_chunk: Input chunk of shape (T, B, d_model)
            offset_t: Time offset for this chunk
        """
        H = x_chunk.mean(dim=1).to(torch.float32).detach()
        T = H.shape[0]
        if T == 0:
            return

        device = H.device
        times = torch.arange(offset_t, offset_t + T, device=device, dtype=torch.float32)
        self._times = torch.cat([self._times.to(device), times], dim=0)
        self._hist  = torch.cat([self._hist.to(device),  H],    dim=0)

        # Trim excess with contiguous memory
        if self._hist.shape[0] > self.depth:
            excess = self._hist.shape[0] - self.depth
            # Use contiguous() to ensure memory efficiency
            self._hist = self._hist[excess:].contiguous().detach()
            self._times = self._times[excess:].contiguous().detach()

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all historical data.

        Returns:
            Tuple of (history, timestamps)
        """
        return self._hist, self._times

# ------------------------------------
# K-L Prior Module (Same as before)
# ------------------------------------

class K_L_Prior_Module(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_components: int = 16, n_tokens: int = 4, tau: float = 64.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components
        self.n_tokens = n_tokens
        self.tau = tau
        self.proj = nn.Linear(n_components * input_dim, n_tokens * output_dim)
        self._cache = {"M": None, "T_hist": 0}

    def compute_global_tokens(self, H_full: torch.Tensor, t_full: torch.Tensor, B: int) -> torch.Tensor:
        device = H_full.device
        T_hist = H_full.shape[0]

        if self.training:
            M = _compute_kl_modes(H_full, t_full, n_components=self.n_components, tau=self.tau, kernel="gauss")
        else:
            if self._cache["T_hist"] == T_hist and self._cache["M"] is not None:
                M = self._cache["M"]
            else:
                M = _compute_kl_modes(H_full, t_full, n_components=self.n_components, tau=self.tau, kernel="gauss")
                if M is not None:
                    self._cache = {"M": M.detach(), "T_hist": T_hist}

        if M is None:
            return torch.zeros(self.n_tokens, B, self.output_dim, device=device)

        M = M.to(self.proj.weight.dtype)
        mem_flat = M.reshape(-1)
        tokens_flat = self.proj(mem_flat)
        tokens = tokens_flat.reshape(self.n_tokens, 1, self.output_dim)
        return tokens.expand(-1, B, -1)

# ------------------------------------
# K-L Internal Memory (Same as before)
# ------------------------------------

class K_L_Internal_Memory(nn.Module):
    def __init__(self, d_model: int, context_window: int, n_components: int = 12, n_tokens: int = 4, tau: float = 12.0):
        super().__init__()
        self.d_model = d_model
        self.context_window = context_window
        self.n_components = n_components
        self.n_tokens = n_tokens
        self.tau = tau
        self.max_hist = 2048 + context_window
        
        self.proj = nn.Linear(n_components * d_model, n_tokens * d_model)

        self.register_buffer("_times", torch.zeros(0, dtype=torch.float32))
        self.register_buffer("_hist",  torch.zeros(0, d_model, dtype=torch.float32))
        self._cache = {"M": None, "T_hist": 0}
        self.register_buffer("_step_counter", torch.tensor([0], dtype=torch.long))

    def reset_state(self, device):
        self._times = torch.zeros(0, device=device, dtype=torch.float32)
        self._hist  = torch.zeros(0, self.d_model, device=device, dtype=torch.float32)
        self._cache = {"M": None, "T_hist": 0}
        self._step_counter.zero_()

    def append_to_history(self, h_t: torch.Tensor, offset_t: int) -> None:
        """
        Append hidden states to internal history buffer.

        Args:
            h_t: Hidden states of shape (T, B, d_model)
            offset_t: Time offset for this chunk
        """
        H = h_t.mean(dim=1).to(torch.float32).detach()
        T = H.shape[0]
        if T == 0:
            return

        device = H.device
        times = torch.arange(offset_t, offset_t + T, device=device, dtype=torch.float32)
        self._times = torch.cat([self._times.to(device), times], dim=0)
        self._hist  = torch.cat([self._hist.to(device),  H],    dim=0)

        # Trim excess with contiguous memory
        if self._hist.shape[0] > self.max_hist:
            excess = self._hist.shape[0] - self.max_hist
            # Use contiguous() to ensure memory efficiency
            self._hist = self._hist[excess:].contiguous().detach()
            self._times = self._times[excess:].contiguous().detach()
            # Invalidate cache when trimming
            self._cache = {"M": None, "T_hist": 0}

    def get_memory_tokens(self, B: int, offset_t: int) -> torch.Tensor:
        """
        Get internal memory tokens from distant history.

        Args:
            B: Batch size
            offset_t: Current time offset

        Returns:
            Memory tokens of shape (n_tokens, B, d_model)
        """
        device = self._hist.device
        self._step_counter[0] += 1

        distant_mask = self._times < (offset_t - self.context_window)
        if distant_mask.sum() < max(self.n_components + 4, 8):
            return torch.zeros(self.n_tokens, B, self.d_model, device=device)

        H = self._hist[distant_mask]
        t = self._times[distant_mask]
        T_hist = H.shape[0]

        M = None
        if self._cache["T_hist"] == T_hist and not self.training:
            M = self._cache["M"]
        elif self._step_counter.item() % CACHE_REFRESH_INTERVAL == 1:
            M = _compute_kl_modes(H, t, n_components=self.n_components, tau=self.tau, kernel="gauss")
            if M is not None and not self.training:
                self._cache = {"M": M.detach(), "T_hist": T_hist}
        else:
            if self._cache["M"] is not None:
                M = self._cache["M"]

        if M is None:
            return torch.zeros(self.n_tokens, B, self.d_model, device=device)

        M = M.to(self.proj.weight.dtype)
        mem_flat = M.reshape(-1)
        tokens_flat = self.proj(mem_flat)
        tokens = tokens_flat.reshape(self.n_tokens, 1, self.d_model)
        return tokens.expand(-1, B, -1)

# ------------------------------------
# NEW: Complete K-L Hybrid Lane
# ------------------------------------

class K_L_Hybrid_Lane(nn.Module):
    """
    Complete lane with ALL components:
    - Mamba Core (fast processor)
    - K-L Dreamer (pilot/world model)
    - K-L Internal (denoised past)
    - K-L RAG (external knowledge)
    """
    def __init__(self, d_model: int, n_ctx: int, out_dim: int = 8, use_rag: bool = True, knowledge_size: int = 100):
        super().__init__()
        self.d_model = d_model
        self.n_ctx = n_ctx
        self.use_rag = use_rag
        self.M_internal = 4  # Internal memory tokens
        self.M_rag = 2  # RAG tokens
        
        # 1. Mamba Core
        self.mamba = MambaCore(d_model, n_layers=2)
        
        # 2. K-L Dreamer (Pilot)
        self.dreamer = K_L_Dreamer(d_model, d_state=32, d_action=4)
        
        # 3. K-L Internal Memory
        self.kl_internal = K_L_Internal_Memory(d_model, context_window=n_ctx, n_tokens=self.M_internal)
        
        # 4. K-L RAG
        if use_rag:
            self.kl_rag = K_L_RAG(d_model, n_tokens=self.M_rag, knowledge_size=knowledge_size)
        else:
            self.kl_rag = None
            
        # Projections
        self.in_proj = nn.Linear(32, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)
        self.output_head = nn.Linear(d_model, out_dim)
        
        self.h_state = None
        
    def reset_state(self, device, B):
        self.h_state = None
        self.kl_internal.reset_state(device)
        self.dreamer.reset_state(B, device)

    def forward(self, x_chunk, offset_t: int, global_tokens=None):
        """
        Complete forward pass with all components.
        """
        B = x_chunk.shape[1]
        T = x_chunk.shape[0]
        device = x_chunk.device

        # 1. Project input
        h_local = self.in_proj(x_chunk)  # (T, B, d_model)
        h_local = self.pos_encoder(h_local, offset=0)
        
        # 2. Get internal memory
        internal_mem = self.kl_internal.get_memory_tokens(B, offset_t)  # (M_internal, B, d_model)
        
        # 3. Prepare input for Mamba: [global_mem, internal_mem, input]
        if global_tokens is not None:
            global_tokens = global_tokens.to(device)
            mamba_input = torch.cat([global_tokens, internal_mem, h_local], dim=0)
            mem_offset = global_tokens.shape[0] + internal_mem.shape[0]
        else:
            mamba_input = torch.cat([internal_mem, h_local], dim=0)
            mem_offset = internal_mem.shape[0]
        
        # 4. Process with Mamba Core
        h_out, self.h_state = self.mamba(mamba_input, self.h_state)
        
        # Extract the output corresponding to actual input
        h_processed = h_out[mem_offset:]  # (T, B, d_model)
        
        # 5. Dreamer analyzes current state
        # Use mean pooled state for policy
        state_summary = h_processed.mean(dim=0)  # (B, d_model)
        dreamer_out = self.dreamer(state_summary, update_state=True)
        
        # 6. Conditionally trigger RAG based on Dreamer's policy
        rag_tokens = None
        if self.kl_rag is not None and self.use_rag:
            trigger_prob = dreamer_out['trigger_rag']  # (B,)
            # During training, use soft gating; during eval, threshold
            if self.training:
                rag_retrieved = self.kl_rag.retrieve(state_summary)  # (M_rag, B, d_model)
                # Soft gate: scale by trigger probability
                trigger_weight = trigger_prob.view(1, -1, 1)  # (1, B, 1)
                rag_tokens = rag_retrieved * trigger_weight  # (M_rag, B, d_model)
            else:
                # Hard threshold during eval with per-batch masking
                should_retrieve = trigger_prob > RAG_TRIGGER_THRESHOLD  # (B,)
                if should_retrieve.any():
                    # Retrieve for all, but mask the output per batch item
                    rag_retrieved = self.kl_rag.retrieve(state_summary)  # (M_rag, B, d_model)
                    # Apply per-batch masking: only apply RAG where triggered
                    mask = should_retrieve.view(1, -1, 1).float()  # (1, B, 1)
                    rag_tokens = rag_retrieved * mask  # (M_rag, B, d_model)

        # 7. Integrate RAG if triggered
        if rag_tokens is not None:
            # Add RAG tokens to processed hidden states
            # Broadcast RAG across time dimension
            rag_broadcasted = rag_tokens.mean(dim=0, keepdim=True).expand(T, -1, -1)  # (T, B, d_model)
            h_processed = h_processed + RAG_INTEGRATION_WEIGHT * rag_broadcasted  # Scaled addition
        
        # 8. Update history
        self.kl_internal.append_to_history(h_processed.detach(), offset_t)
        
        # 9. Output
        output = self.output_head(h_processed)
        
        return output, dreamer_out

# ------------------------------------
# COMPLETE: Hierarchical Ensemble
# ------------------------------------

class HierarchicalEnsemble(nn.Module):
    """
    COMPLETE architecture with all components from the diagram.
    """
    def __init__(self, d_model, n_ctx, out_dim, use_rag=True, knowledge_size=100):
        super().__init__()
        self.d_model = d_model
        self.n_ctx = n_ctx
        self.out_dim = out_dim
        self.use_rag = use_rag

        # Global History
        self.global_history = GlobalHistoryBuffer(depth=5000, d_model=32)

        # K-L Prior (Global Denoiser)
        self.prior_kl_module = K_L_Prior_Module(
            input_dim=32, 
            output_dim=d_model, 
            n_components=16, 
            n_tokens=4,  # FIXED: Match internal memory
            tau=64.0
        )

        # Two Complete Hybrid Lanes
        self.lane_1 = K_L_Hybrid_Lane(d_model, n_ctx, out_dim, use_rag, knowledge_size)
        self.lane_2 = K_L_Hybrid_Lane(d_model, n_ctx, out_dim, use_rag, knowledge_size)

        # Tracking
        self.trigger_count_1 = 0
        self.trigger_count_2 = 0

    def reset_state(self, device, B):
        self.lane_1.reset_state(device, B)
        self.lane_2.reset_state(device, B)
        self.trigger_count_1 = 0
        self.trigger_count_2 = 0
        
    def forward(self, x_chunk, offset_t: int):
        """
        Complete forward with world model losses.
        """
        B = x_chunk.shape[1]
        device = x_chunk.device

        # 1. Append to global history
        self.global_history.append(x_chunk, offset_t)

        # 2. Get global tokens from K-L Prior
        raw_hist, raw_times = self.global_history.get_all()
        global_tokens = None
        if raw_hist.shape[0] >= 16:
            global_tokens = self.prior_kl_module.compute_global_tokens(raw_hist, raw_times, B)
        
        # 3. Process through both lanes
        output_1, dreamer_out_1 = self.lane_1(x_chunk, offset_t, global_tokens)
        output_2, dreamer_out_2 = self.lane_2(x_chunk, offset_t, global_tokens)
        
        # 4. Aggregate outputs
        final_output = (output_1 + output_2) / 2.0
        
        # 5. Track triggers
        if dreamer_out_1['trigger_rag'].mean() > RAG_TRIGGER_THRESHOLD:
            self.trigger_count_1 += 1
        if dreamer_out_2['trigger_rag'].mean() > RAG_TRIGGER_THRESHOLD:
            self.trigger_count_2 += 1
        
        # 6. Return outputs and auxiliary losses
        aux_losses = {
            'dreamer_1': dreamer_out_1,
            'dreamer_2': dreamer_out_2,
        }
        
        return final_output, aux_losses

# ------------------------------------
# Training Loop
# ------------------------------------

def make_long_noisy_song(
    T: int = 2000,
    B: int = 16,
    F_in: int = 32,
    F_out: int = 8,
    device: str = "cpu",
    seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic noisy sequence data for training.

    Args:
        T: Sequence length
        B: Batch size
        F_in: Input feature dimension
        F_out: Output feature dimension
        device: Device to create tensors on
        seed: Random seed

    Returns:
        Tuple of (input_sequence, target_sequence)
    """
    g = torch.Generator(device=device).manual_seed(seed)
    t = torch.arange(T, device=device).float().view(T, 1, 1)
    freqs = torch.logspace(-1.5, 0.0, F_in // 2, device=device).view(1, 1, -1)
    signal = torch.cat([torch.sin(t*freqs), torch.cos(t*freqs)], dim=-1).repeat(1, B, 1)
    noise = torch.randn(T, B, F_in, generator=g, device=device) * 1.2
    X = signal + noise
    W = torch.randn(F_in, F_out, generator=g, device=device) * 0.5
    Y = torch.einsum("tbf,fo->tbo", signal, W)
    return X.to(torch.float32), Y.to(torch.float32)

def train_loop(
    device: torch.device,
    epochs: int = 50,
    T: int = 2000,
    B: int = 16,
    d_model: int = 64,
    n_ctx: int = 256,
    lr: float = 1e-3,
    use_rag: bool = True,
    knowledge_size: int = 100
) -> HierarchicalEnsemble:
    """
    Main training loop for the K-L Prior Ensemble.

    Args:
        device: Device to train on
        epochs: Number of training epochs
        T: Total sequence length
        B: Batch size
        d_model: Model dimension
        n_ctx: Context window size
        lr: Learning rate
        use_rag: Whether to enable RAG modules
        knowledge_size: Size of RAG knowledge base

    Returns:
        Trained model
    """
    X, Y = make_long_noisy_song(T=T, B=B, F_in=32, F_out=8, device=device)
    
    model = HierarchicalEnsemble(
        d_model=d_model, 
        n_ctx=n_ctx, 
        out_dim=8,
        use_rag=use_rag,
        knowledge_size=knowledge_size
    ).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\n{'='*60}")
    print(f"🚀 COMPLETE K-L Prior Ensemble Training")
    print(f"{'='*60}")
    print(f"✅ Prior K-L Module (Global Denoiser)")
    print(f"✅ Mamba Core ({'Real SSM' if model.lane_1.mamba.is_mamba else 'GRU Fallback'})")
    print(f"✅ K-L Dreamer (World Model Pilots)")
    print(f"✅ K-L Internal Memory")
    print(f"✅ K-L RAG (External Knowledge)" if use_rag else "❌ RAG Disabled")
    print(f"{'='*60}")
    print(f"Device: {device} | Epochs: {epochs} | Context: {n_ctx} | d_model: {d_model}")
    print(f"{'='*60}\n")
    
    for ep in range(1, epochs+1):
        model.train()
        model.reset_state(device, B)
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_reward_loss = 0.0
        chunks = 0
        
        for t_start in range(0, T, n_ctx):
            x_chunk = X[t_start : t_start + n_ctx]
            y_chunk = Y[t_start : t_start + n_ctx]
            
            if x_chunk.shape[0] == 0: 
                continue
                
            # Forward pass
            yhat, aux_losses = model(x_chunk, offset_t=t_start)
            
            # Main prediction loss
            pred_loss = F.mse_loss(yhat, y_chunk)
            
            # World model losses (optional, can be weighted down)
            recon_loss_1 = F.mse_loss(
                aux_losses['dreamer_1']['obs_recon'], 
                model.lane_1.in_proj(x_chunk).mean(dim=0)
            )
            recon_loss_2 = F.mse_loss(
                aux_losses['dreamer_2']['obs_recon'], 
                model.lane_2.in_proj(x_chunk).mean(dim=0)
            )
            recon_loss = (recon_loss_1 + recon_loss_2) / 2.0
            
            # Reward prediction loss (dummy target: negative prediction error)
            with torch.no_grad():
                reward_target = -F.mse_loss(yhat, y_chunk, reduction='none').mean(dim=(0, 2))
            reward_loss_1 = F.mse_loss(aux_losses['dreamer_1']['reward_pred'], reward_target)
            reward_loss_2 = F.mse_loss(aux_losses['dreamer_2']['reward_pred'], reward_target)
            reward_loss = (reward_loss_1 + reward_loss_2) / 2.0
            
            # Total loss with weighted auxiliary losses
            loss = pred_loss + RECON_LOSS_WEIGHT * recon_loss + REWARD_LOSS_WEIGHT * reward_loss
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            opt.step()

            total_loss += pred_loss.item()
            total_recon_loss += recon_loss.item()
            total_reward_loss += reward_loss.item()
            chunks += 1

        avg_loss = total_loss / max(1, chunks)
        avg_recon = total_recon_loss / max(1, chunks)
        avg_reward = total_reward_loss / max(1, chunks)
        
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"[Epoch {ep:03d}] "
                  f"MSE={avg_loss:.4f} | "
                  f"Recon={avg_recon:.4f} | "
                  f"Reward={avg_reward:.4f} | "
                  f"RAG_L1={model.trigger_count_1} | "
                  f"RAG_L2={model.trigger_count_2}")
    
    # Final eval
    print(f"\n{'='*60}")
    print("📊 Final Evaluation")
    print(f"{'='*60}")
    model.eval()
    with torch.no_grad():
        model.reset_state(device, B)
        mse_sum = 0.0
        chunks = 0
        for t_start in range(0, T, n_ctx):
            x = X[t_start : t_start+n_ctx]
            y = Y[t_start : t_start+n_ctx]
            
            if x.shape[0] == 0: 
                continue
            
            yhat, _ = model(x, offset_t=t_start)
            mse_sum += F.mse_loss(yhat, y).item()
            chunks += 1
            
        final_mse = mse_sum / max(1, chunks)
        print(f"✅ Final MSE: {final_mse:.4f}")
        print(f"✅ Lane 1 RAG Triggers: {model.trigger_count_1}")
        print(f"✅ Lane 2 RAG Triggers: {model.trigger_count_2}")
        print(f"{'='*60}\n")
        
    return model

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="COMPLETE: K-L Prior Ensemble with Dreamer + RAG")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--use-rag", action='store_true', help="Enable RAG modules")
    ap.add_argument("--knowledge-size", type=int, default=100, help="Size of RAG knowledge base")
    args = ap.parse_args()

    # Validate inputs
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if args.ctx <= 0:
        raise ValueError(f"ctx must be positive, got {args.ctx}")
    if args.d_model <= 0:
        raise ValueError(f"d_model must be positive, got {args.d_model}")
    if args.knowledge_size <= 0:
        raise ValueError(f"knowledge_size must be positive, got {args.knowledge_size}")
    if args.lr <= 0:
        raise ValueError(f"lr must be positive, got {args.lr}")

    device = torch.device(args.device)
    logger.info(f"Starting training with device: {device}")

    train_loop(
        device=device,
        epochs=args.epochs,
        B=16,
        d_model=args.d_model,
        n_ctx=args.ctx,
        lr=args.lr,
        use_rag=args.use_rag,
        knowledge_size=args.knowledge_size
    )

if __name__ == "__main__":
    main()
