# Predictive Coding MLP Block + Language Head for Transformer

## Overview

2-layer Predictive Coding MLP block and Language Head for transformer decoder integration. Uses error-driven learning with ngclearn framework.

## Architecture

```
Attention Output [B, T, d] → [Pre-Norm] → MLP (z_mlp1 → z_mlp2) → [Residual] → [Post-Norm] → Language Head (z_out) → Vocab Logits [B, T, V]
```

**Components:**
- **MLP Block**: Expansion (d → d_ff) then contraction (d_ff → d) Note: d_ff = 4*d
- **Language Head**: Projects to vocabulary size V
- **Normalization**: Optional pre-norm and post-norm
- **Residual**: Optional skip connection

## Functions

### `forward(attention_output)`
**Inference** - processes input without weight updates.
- Returns: Vocabulary logits `[B, T, V]`
- Use for: Predictions, evaluation, inference

### `train_step(attention_output, target_logits=None)`
**Training** - processes input and updates weights.
- Returns: (vocabulary logits, EFE loss)
- Use for: Training with target labels

### When to Use

**Use `forward()` when:**
- Making predictions
- Evaluating the model
- Inference after training
- No labels available

**Use `train_step()` when:**
- Training the model
- You have target labels
- You need loss for backpropagation
- You want to update weights

## Dependencies

```bash
pip install jax jaxlib ngclearn
```

**Required:**
- Python 3.8+
- JAX (for numerical computing)
- ngclearn (Neural Generative Coding framework)

## Usage

```python
from jax import random
from mlp import SimplePCNMLP

# Initialize
pcn_mlp = SimplePCNMLP(
    random.PRNGKey(0),
    input_dim=768,
    hidden1_dim=3072,
    vocab_size=50257
)

# Inference
vocab_logits = pcn_mlp.forward(attention_output)

# Training
vocab_logits, loss = pcn_mlp.train_step(attention_output, target_logits)
```

