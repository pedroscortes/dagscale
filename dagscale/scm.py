"""Random linear-Gaussian SCMs X = W X + eps on Erdos-Renyi and scale-free DAGs; W[i, j] is the weight of j -> i."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class LinearGaussianSCM:
    weights: jax.Array
    noise_scale: jax.Array

    def sample(self, key: jax.Array, n: int) -> jax.Array:
        d = self.weights.shape[0]
        omega = jnp.linalg.solve(jnp.eye(d) - self.weights, jnp.eye(d))
        noise = jax.random.normal(key, shape=(n, d)) * self.noise_scale
        return noise @ omega.T

    @classmethod
    def random(cls, d, *, edge_probability=0.3, weight_range=(0.5, 2.0), noise_range=(0.5, 1.5), seed=0):
        """Erdos-Renyi DAG over a random order; weights uniform in +-weight_range, noise sd uniform in noise_range."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(d)
        rank = np.empty(d, dtype=int)
        rank[perm] = np.arange(d)
        parent_mask = rank[None, :] < rank[:, None]
        edge_mask = rng.uniform(size=(d, d)) < edge_probability
        signs = rng.choice(np.array([-1.0, 1.0]), size=(d, d))
        magnitudes = rng.uniform(*weight_range, size=(d, d))
        weights = np.where(parent_mask & edge_mask, signs * magnitudes, 0.0)
        return cls(weights=jnp.asarray(weights), noise_scale=jnp.asarray(rng.uniform(*noise_range, size=d)))

    @classmethod
    def random_scale_free(cls, d, *, m=2, weight_range=(0.5, 2.0), noise_range=(0.5, 1.5), seed=0):
        """Preferential attachment in node order: node i takes min(m, i) parents, chosen with weight out-degree + 1."""
        rng = np.random.default_rng(seed)
        weights = np.zeros((d, d))
        out_degree = np.zeros(d, dtype=int)
        for i in range(1, d):
            if i <= m:
                parents = np.arange(i)
            else:
                probs = out_degree[:i].astype(float) + 1.0
                probs /= probs.sum()
                parents = rng.choice(i, size=m, replace=False, p=probs)
            for parent in parents:
                sign = rng.choice(np.array([-1.0, 1.0]))
                weights[i, int(parent)] = sign * rng.uniform(*weight_range)
                out_degree[int(parent)] += 1
        return cls(weights=jnp.asarray(weights), noise_scale=jnp.asarray(rng.uniform(*noise_range, size=d)))
