# CLAUDE.md - AI Assistant Guide

**Last Updated**: 2025-11-15
**Project**: K-L Prior Ensemble with Dreamer + RAG + Mamba SSM
**Version**: 1.0

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Architecture Guide](#architecture-guide)
4. [Development Workflows](#development-workflows)
5. [Code Conventions](#code-conventions)
6. [Known Issues](#known-issues)
7. [Testing Strategy](#testing-strategy)
8. [AI Assistant Guidelines](#ai-assistant-guidelines)

---

## Project Overview

### What is This Project?

This is a research implementation of a **Hierarchical K-L Prior Ensemble** that combines:
- **K-L Prior Decomposition**: Karhunen-Loève expansion for temporal denoising
- **DreamerV3-style World Model**: For predictive learning and policy control
- **RAG (Retrieval-Augmented Generation)**: External knowledge integration
- **Mamba SSM**: State Space Model for efficient sequence processing (with GRU fallback)

### Project Purpose

The system processes long noisy sequences and learns to extract clean signals through:
1. **Global denoising** via K-L Prior on long-term history
2. **Local processing** via Mamba/SSM with sliding windows
3. **Predictive modeling** via Dreamer world model
4. **Knowledge retrieval** via RAG when needed

### Tech Stack

- **Language**: Python 3
- **Framework**: PyTorch
- **Optional Dependencies**: `mamba-ssm` (falls back to GRU if not available)
- **Device Support**: CPU and CUDA

---

## Codebase Structure

### File Organization

```
/
├── complete_solution.py    # Main implementation (872 lines)
├── CODE_REVIEW.md          # Comprehensive code review with known issues
└── CLAUDE.md               # This file
```

### Module Architecture (complete_solution.py)

The code is organized into logical sections:

```
complete_solution.py
│
├── K-L Math Components (Lines 27-95)
│   ├── positional_encoding()          # Positional encoding generation
│   ├── _time_kernel()                 # Temporal kernel computation
│   ├── _compute_kl_modes()            # K-L mode extraction
│   └── PositionalEncoding (nn.Module) # Positional encoding module
│
├── K-L RAG Module (Lines 98-169)
│   └── K_L_RAG (nn.Module)            # External knowledge retrieval
│
├── K-L Dreamer (Lines 172-306)
│   └── K_L_Dreamer (nn.Module)        # World model pilot
│       ├── Encoder/Decoder
│       ├── RSSM (GRU-based)
│       ├── Policy Network
│       ├── Value Network
│       └── Reward Predictor
│
├── Mamba Core (Lines 309-372)
│   └── MambaCore (nn.Module)          # Real Mamba SSM or GRU fallback
│
├── Memory Management (Lines 375-518)
│   ├── GlobalHistoryBuffer            # Long-term global history
│   ├── K_L_Prior_Module               # Global denoiser
│   └── K_L_Internal_Memory            # Lane-specific memory
│
├── Complete Architecture (Lines 521-710)
│   ├── K_L_Hybrid_Lane                # Single processing lane
│   └── HierarchicalEnsemble           # Top-level ensemble of 2 lanes
│
└── Training & CLI (Lines 713-871)
    ├── make_long_noisy_song()         # Synthetic data generation
    ├── train_loop()                   # Training procedure
    └── main()                         # CLI entry point
```

---

## Architecture Guide

### Component Dependencies

```
HierarchicalEnsemble
├── GlobalHistoryBuffer (shared)
├── K_L_Prior_Module (global denoiser, shared)
├── K_L_Hybrid_Lane #1
│   ├── MambaCore
│   ├── K_L_Dreamer
│   ├── K_L_Internal_Memory
│   └── K_L_RAG (optional)
└── K_L_Hybrid_Lane #2
    ├── MambaCore
    ├── K_L_Dreamer
    ├── K_L_Internal_Memory
    └── K_L_RAG (optional)
```

### Data Flow

1. **Input**: Noisy sequence chunks `(T, B, F_in)` where:
   - `T` = sequence length
   - `B` = batch size
   - `F_in` = input features (default: 32)

2. **Global Processing**:
   - Raw input appended to `GlobalHistoryBuffer`
   - K-L Prior extracts global denoised tokens from full history
   - Global tokens shared across both lanes

3. **Per-Lane Processing**:
   - Each lane maintains its own `K_L_Internal_Memory`
   - Mamba processes: `[global_tokens, internal_memory, input_chunk]`
   - Dreamer analyzes state and decides whether to trigger RAG
   - RAG retrieves external knowledge if triggered
   - Output prediction generated

4. **Output**: Two predictions averaged together

### Critical State Management

Each lane maintains stateful components:
- **MambaCore**: Hidden state `h_state` (for GRU) or internal state (for Mamba)
- **K_L_Dreamer**: Dream state `h_dream`
- **K_L_Internal_Memory**: History buffer `(_hist, _times)` + cache

**Important**: Must call `reset_state()` before processing new sequences!

---

## Development Workflows

### Running the Code

```bash
# Basic training (GRU fallback)
python3 complete_solution.py --epochs 50

# With RAG enabled
python3 complete_solution.py --epochs 50 --use-rag --knowledge-size 100

# Custom hyperparameters
python3 complete_solution.py \
    --epochs 100 \
    --ctx 512 \
    --d_model 128 \
    --lr 5e-4 \
    --device cuda \
    --use-rag \
    --knowledge-size 200
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | str | auto | Device: "cpu" or "cuda" |
| `--epochs` | int | 50 | Number of training epochs |
| `--ctx` | int | 256 | Context window size |
| `--lr` | float | 1e-3 | Learning rate |
| `--d_model` | int | 64 | Model dimension |
| `--use-rag` | flag | False | Enable RAG modules |
| `--knowledge-size` | int | 100 | Size of RAG knowledge base |

### Adding New Features

When extending the code:

1. **Add new memory component**:
   - Follow pattern of `K_L_Internal_Memory`
   - Use `register_buffer()` for persistent state
   - Implement `reset_state()` method
   - Add to lane's `reset_state()` call chain

2. **Add new lane component**:
   - Add to `K_L_Hybrid_Lane.__init__()`
   - Integrate in `forward()` pass
   - Update auxiliary loss computation if needed

3. **Modify architecture**:
   - Update both lanes symmetrically
   - Ensure state management is correct
   - Test with `reset_state()` calls

---

## Code Conventions

### PyTorch Conventions

1. **Tensor Shape Convention**: `(T, B, D)` format (time-first)
   - `T` = sequence length
   - `B` = batch size
   - `D` = feature dimension

2. **Device Management**:
   - Use `.to(device)` sparingly
   - Prefer registering parameters/buffers properly
   - All module params should share device

3. **State Buffers**:
   - Use `register_buffer()` for non-learnable state
   - Use `nn.Parameter()` for learnable state
   - Always call `.detach()` when storing state

### Numerical Stability Practices

1. **Epsilon Handling**:
   ```python
   # Good: explicit epsilon
   tau = max(float(tau), 1e-12)

   # Good: adaptive epsilon based on size
   eps = 1e-8 if T <= 2048 else 1e-6
   ```

2. **Normalization**:
   ```python
   # Good: normalize eigenvectors
   phi = phi / (phi.norm(dim=0, keepdim=True) + 1e-12)
   ```

3. **Kernel Symmetry**:
   ```python
   # Good: enforce symmetry
   K = 0.5 * (K + K.T)
   ```

### Memory Management

1. **Detach for Storage**:
   ```python
   # Good: detach when storing
   self.h_dream = next_state.detach()

   # Good: detach history
   H = h_t.mean(dim=1).to(torch.float32).detach()
   ```

2. **Buffer Trimming**:
   ```python
   # Trim old history when exceeding capacity
   if self._hist.shape[0] > self.max_hist:
       excess = self._hist.shape[0] - self.max_hist
       self._hist = self._hist[excess:].detach()
   ```

### Coding Style

1. **Naming Conventions**:
   - Classes: `PascalCase` (e.g., `K_L_Prior_Module`)
   - Functions: `snake_case` (e.g., `compute_kl_modes`)
   - Private helpers: `_leading_underscore` (e.g., `_time_kernel`)
   - Tensors: lowercase with semantic names (e.g., `h_local`, `tokens`)

2. **Magic Numbers**:
   - **Current state**: Magic numbers scattered throughout
   - **Best practice**: Extract to named constants
   - **When modifying**: Consider refactoring to constants

3. **Comments**:
   - Use section headers: `# === Section Name ===`
   - Use inline comments for non-obvious operations
   - Docstrings for public APIs

---

## Known Issues

**CRITICAL**: See `CODE_REVIEW.md` for comprehensive issue list. Key issues:

### Priority 1 (Fix Before Use)

1. **Index Out of Bounds** (`complete_solution.py:93-95`)
   - `PositionalEncoding.forward()` can crash on long sequences
   - Add bounds checking before production use

2. **Unsafe Exception Handling** (`complete_solution.py:68-75`)
   - Catches all exceptions including KeyboardInterrupt
   - No logging makes debugging impossible
   - Refactor to catch specific exceptions

3. **Incorrect RAG Masking** (`complete_solution.py:614-617`)
   - RAG applied to entire batch even when only some items should retrieve
   - Implement per-sample masking

### Priority 2 (Performance)

4. **Inefficient Loop** (`complete_solution.py:156-159`)
   - Nested Python loop in RAG retrieval (10-100x slower than vectorized)
   - Vectorize using `torch.gather()` or index operations

5. **Memory Leak Potential** (`complete_solution.py:387-401`)
   - Repeated concatenation creates new tensors
   - Consider circular buffer implementation

### Code Quality Issues

- No input validation on CLI arguments
- Magic numbers throughout (e.g., `0.1`, `0.05`, `25`)
- Missing type hints on many methods
- Inconsistent documentation
- No unit tests

---

## Testing Strategy

### Current State

**Test Coverage**: 0% (no tests exist)

### Recommended Test Structure

```
tests/
├── test_kl_math.py
│   ├── test_positional_encoding()
│   ├── test_time_kernel()
│   └── test_compute_kl_modes()
├── test_components.py
│   ├── test_rag_retrieval()
│   ├── test_dreamer_forward()
│   ├── test_mamba_core()
│   └── test_memory_buffers()
├── test_integration.py
│   ├── test_lane_forward()
│   ├── test_ensemble_forward()
│   └── test_training_loop()
└── test_edge_cases.py
    ├── test_empty_sequences()
    ├── test_long_sequences()
    └── test_device_transfers()
```

### Critical Tests to Add

1. **Bounds Checking**:
   ```python
   def test_positional_encoding_bounds():
       pe = PositionalEncoding(d_model=64, max_len=100)
       x = torch.randn(150, 8, 64)  # Exceeds max_len
       # Should raise error or handle gracefully
   ```

2. **State Reset**:
   ```python
   def test_state_reset():
       model = HierarchicalEnsemble(...)
       model.reset_state(device, B=16)
       # Verify all states are zeroed
   ```

3. **Gradient Flow**:
   ```python
   def test_gradients_flow():
       model = HierarchicalEnsemble(...)
       output, aux = model(x_chunk, offset_t=0)
       loss = output.sum()
       loss.backward()
       # Verify all parameters have gradients
   ```

---

## AI Assistant Guidelines

### When Making Changes

1. **Always Read CODE_REVIEW.md First**
   - Understand known issues before modifying
   - Don't introduce new instances of known problems
   - Reference issue numbers when fixing

2. **Maintain Architectural Consistency**
   - Both lanes should be symmetric
   - State management must be consistent
   - Don't break the `(T, B, D)` convention

3. **Test State Management**
   - Always test with `reset_state()` calls
   - Verify no state leaks between sequences
   - Check device consistency

4. **Preserve Numerical Stability**
   - Keep epsilon handling
   - Maintain kernel symmetry
   - Test with different tensor sizes

### Adding Features Checklist

- [ ] Does it maintain `(T, B, D)` tensor format?
- [ ] Does it handle device transfers correctly?
- [ ] Does it implement `reset_state()` if stateful?
- [ ] Does it use `register_buffer()` for persistent state?
- [ ] Does it detach tensors when storing state?
- [ ] Does it add corresponding tests?
- [ ] Does it update both lanes if architectural?
- [ ] Does it handle edge cases (empty tensors, long sequences)?

### Refactoring Guidelines

When refactoring:

1. **Extract Magic Numbers First**
   - Replace with named constants
   - Document the meaning
   - Make configurable if appropriate

2. **Add Type Hints**
   - Use `torch.Tensor` for tensors
   - Use `Optional[T]` for nullable returns
   - Document tensor shapes in docstrings

3. **Vectorize Loops**
   - Replace Python loops with tensor operations
   - Use `torch.einsum()` for complex operations
   - Profile before and after

4. **Improve Error Handling**
   - Replace bare `except:` with specific exceptions
   - Add logging instead of silent failures
   - Validate inputs at boundaries

### Common Pitfalls to Avoid

1. **Don't Break State Management**
   ```python
   # BAD: Forgetting to detach
   self.h_state = next_state  # Keeps gradient graph

   # GOOD: Detach when storing
   self.h_state = next_state.detach()
   ```

2. **Don't Mix Tensor Formats**
   ```python
   # BAD: Inconsistent format
   x = x.permute(1, 0, 2)  # Now (B, T, D)
   # ... many operations later ...
   # Code expects (T, B, D) - BUG!

   # GOOD: Keep format consistent or document changes
   ```

3. **Don't Ignore Device Mismatches**
   ```python
   # BAD: Assuming device
   tokens = torch.zeros(M, B, D)  # Defaults to CPU

   # GOOD: Explicit device
   tokens = torch.zeros(M, B, D, device=device)
   ```

4. **Don't Skip Validation**
   ```python
   # BAD: No validation
   def forward(self, x):
       return self.proj(x)

   # GOOD: Validate shapes
   def forward(self, x):
       assert x.shape[-1] == self.input_dim
       return self.proj(x)
   ```

### Performance Optimization Guidelines

1. **Profile Before Optimizing**
   - Use `torch.profiler` to find bottlenecks
   - Don't optimize without measurements
   - Document performance improvements

2. **Vectorization Priority**
   - Level 1: Remove Python loops (highest impact)
   - Level 2: Reduce device transfers
   - Level 3: Optimize memory access patterns
   - Level 4: Use fused operations

3. **Memory Optimization**
   - Use `torch.no_grad()` for inference
   - Clear caches periodically
   - Use `del` for large intermediate tensors
   - Consider gradient checkpointing for long sequences

### Debugging Tips

1. **State Inspection**
   ```python
   # Add temporary debugging
   print(f"h_state shape: {self.h_state.shape if self.h_state is not None else None}")
   print(f"h_state device: {self.h_state.device if self.h_state is not None else None}")
   ```

2. **Gradient Debugging**
   ```python
   # Check for NaN gradients
   for name, param in model.named_parameters():
       if param.grad is not None:
           if torch.isnan(param.grad).any():
               print(f"NaN gradient in {name}")
   ```

3. **Shape Debugging**
   ```python
   # Use descriptive assertions
   assert x.shape == (T, B, self.d_model), \
       f"Expected (T={T}, B={B}, D={self.d_model}), got {x.shape}"
   ```

### Documentation Standards

When documenting code:

1. **Module Docstrings**:
   ```python
   class NewComponent(nn.Module):
       """
       One-line summary.

       Detailed explanation of what this component does,
       its role in the architecture, and key behaviors.

       Args:
           d_model: Model dimension
           n_tokens: Number of output tokens

       Attributes:
           proj: Projection layer
           state: Internal state buffer

       Example:
           >>> comp = NewComponent(d_model=64, n_tokens=4)
           >>> output = comp(input_tensor)
       """
   ```

2. **Function Docstrings**:
   ```python
   def compute_something(
       tensor: torch.Tensor,
       param: float
   ) -> torch.Tensor:
       """
       Compute something from tensor.

       Args:
           tensor: Input tensor of shape (T, B, D)
           param: Control parameter in range [0, 1]

       Returns:
           Processed tensor of shape (T, B, D)

       Raises:
           ValueError: If param not in valid range
       """
   ```

3. **Inline Comments**:
   ```python
   # Good: Explain WHY, not WHAT
   K = 0.5 * (K + K.T)  # Enforce symmetry for numerical stability

   # Bad: Obvious statement
   K = 0.5 * (K + K.T)  # Average K with its transpose
   ```

---

## Quick Reference

### File Line Number Reference

| Component | Lines | Description |
|-----------|-------|-------------|
| Positional Encoding | 30-95 | PE generation and module |
| K-L RAG | 98-169 | External knowledge retrieval |
| K-L Dreamer | 172-306 | World model pilot |
| MambaCore | 309-372 | SSM/GRU processor |
| GlobalHistoryBuffer | 375-404 | Shared history |
| K_L_Prior_Module | 407-442 | Global denoiser |
| K_L_Internal_Memory | 445-518 | Lane memory |
| K_L_Hybrid_Lane | 521-632 | Complete lane |
| HierarchicalEnsemble | 635-710 | Top-level model |
| Training | 713-843 | Train loop |
| CLI | 846-871 | Main entry |

### Key Constants

```python
# Default hyperparameters
D_MODEL = 64           # Model dimension
N_CTX = 256            # Context window
N_COMPONENTS_GLOBAL = 16    # Global K-L components
N_COMPONENTS_INTERNAL = 12  # Internal K-L components
TAU_GLOBAL = 64.0      # Global kernel bandwidth
TAU_INTERNAL = 12.0    # Internal kernel bandwidth
GLOBAL_DEPTH = 5000    # Global history size
INTERNAL_MAX = 2048    # Internal history size
```

### State Management Checklist

Before running inference on new sequence:
```python
model.reset_state(device, B)
```

This resets:
- [ ] `lane_1.h_state` (Mamba state)
- [ ] `lane_1.kl_internal._hist` (internal memory)
- [ ] `lane_1.dreamer.h_dream` (dream state)
- [ ] `lane_2.h_state` (Mamba state)
- [ ] `lane_2.kl_internal._hist` (internal memory)
- [ ] `lane_2.dreamer.h_dream` (dream state)
- [ ] Trigger counters

Note: `GlobalHistoryBuffer` is NOT reset (accumulates across sequences).

---

## Version History

- **v1.0** (2025-11-15): Initial CLAUDE.md created
  - Documented complete architecture
  - Added known issues from CODE_REVIEW.md
  - Established conventions and guidelines

---

## Additional Resources

- **CODE_REVIEW.md**: Comprehensive code review with detailed issue analysis
- **complete_solution.py**: Main implementation with inline comments

---

**End of CLAUDE.md**
