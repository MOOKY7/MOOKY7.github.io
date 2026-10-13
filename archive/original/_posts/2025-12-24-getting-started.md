---
layout: post
title: "Getting started with JAX: jit, grad, and vmap"
tags: [jax, python, autodiff]
---

This is my first post on this site. I’m learning **JAX** because it makes numerical computing feel composable:
you write normal Python/NumPy-style code, then you can *transform* it (differentiate, compile, vectorize).

## What I want from JAX

- Fast array code (XLA compilation)
- Automatic differentiation that works with my functions
- Easy batching (vectorization) without writing loops
- A clean path toward research code (and eventually bigger models)

## The mental model: write functions, then transform them

In JAX, you often:
1) write a pure function, and then
2) apply transforms like `grad`, `jit`, and `vmap`.

### `grad`: automatic differentiation

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(jnp.sin(x) * x**2)

dfdx = jax.grad(f)

x = jnp.array([1.0, 2.0, 3.0])
print(dfdx(x))

'''