---
{
  "title": "Getting started with JAX",
  "date": "2025-12-24",
  "updated": "2026-10-12",
  "slug": "getting-started-with-jax",
  "track": "architecture",
  "kind": "Code note",
  "summary": "Start with a small numerical function. Differentiate it, apply it across a batch, then compile it.",
  "tags": ["JAX", "Python", "Autodiff"],
  "published": true
}
---

I am learning JAX because its function transformations give me a direct route from a mathematical expression to executable code. This note updates the short introduction from my original site.

## Start with a function

The example below maps a vector to one number. Its derivative with respect to element i is `cos(x_i) * x_i**2 + 2 * x_i * sin(x_i)`.

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(jnp.sin(x) * x**2)

x = jnp.array([1.0, 2.0, 3.0])
gradient = jax.grad(f)(x)
expected = jnp.cos(x) * x**2 + 2 * x * jnp.sin(x)
assert jnp.allclose(gradient, expected, rtol=1e-5)
```

## Compose the transformations

`grad` creates a gradient function. `vmap` applies a function across a batch. `jit` compiles a function for execution. These operations can be composed.

```python
batch = jnp.stack([x, 2 * x])
batch_grad = jax.jit(jax.vmap(jax.grad(f)))
result = batch_grad(batch)
assert result.shape == batch.shape
```

## What I would check next

Before timing a larger example, check the numerical output and array shapes. When measuring execution time, distinguish compilation from later calls and wait for asynchronous work to finish.

## Source

[JAX transformations documentation](https://docs.jax.dev/en/latest/101/transformations.html) and [JAX quickstart](https://docs.jax.dev/en/latest/quickstart.html).
