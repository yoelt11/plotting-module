from flax.struct import dataclass
import jax.numpy as jnp
import numpy as np

@dataclass
class Lambdas:
    lambdas: jnp.ndarray # shape (N_KERNELS, N_PARAMETERS), where N_parameters = 6, and parameters: [mu_x, mu_y, sigma_x, sigma_y, theta]
    x: jnp.ndarray # shape (N_POINTS,), where N_POINTS is the number of points in the x-axis
    y: jnp.ndarray # shape (N_POINTS,), where N_POINTS is the number of points in the y-axis

    def __post_init__(self):
        if len(self.lambdas.shape) != 2 or self.lambdas.shape[1] != 6:
            raise ValueError(
                f"Expected lambdas to have shape (N_kernels, 6), but got {self.lambdas.shape}"
            )

@dataclass
class Solution:
    values: jnp.ndarray
    x: jnp.ndarray
    y: jnp.ndarray
    legend: str

@dataclass
class Curve:
    values: jnp.ndarray
    items: jnp.ndarray
    legend: str

@dataclass
class CurveWithStd:
    mean: jnp.ndarray
    std: jnp.ndarray
    items: jnp.ndarray
    legend: str
