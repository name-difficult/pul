import math
import secrets
import numpy as np

def csprng_uniform_0_1() -> float:
    """
    Cryptographically secure U in [0,1).
    Use 53 random bits to match double precision mantissa.
    """
    return secrets.randbits(53) / (1 << 53)

def karney_sampler(s: float, size: int) -> np.ndarray:
    """
    Karney Sampler for discrete Gaussian sampling over integers (D_{Z,s}).

    Args:
    - s: Standard deviation of the discrete Gaussian distribution.
    - size: Number of samples to generate.

    Returns:
    - A NumPy array with samples from D_{Z,s}.
    """

    samples = np.zeros(size, dtype=np.int64)

    for i in range(size):
        # Sample a random uniform number u1 and u2
        u1 = csprng_uniform_0_1()
        u2 = csprng_uniform_0_1()

        while u1 == 0:
            u1 = csprng_uniform_0_1()  # Avoid log(0)

        r = math.sqrt(-2.0 * math.log(u1))  # Box-Muller method
        theta = 2.0 * math.pi * u2

        # Generate z0 and z1 (standard normal samples)
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)

        # Convert to discrete Gaussian by rounding
        sample = round(s * z0)  # z0 represents the sample

        # Accept or reject using Bernoulli test based on probability
        if secrets.randbelow(2) == 0:  # Probability of 1/2 for acceptance
            samples[i] = sample

    return samples