from hdc.hv import HDCRepresentation
from typing import Type
import numpy as np
from hdc.encoder import Encoder

class LevelEncoder(Encoder):
    def __init__(self, hdc: Type[HDCRepresentation], N, min, max, step) -> None:
        self.hdc = hdc
        self.N = N
        self.min = min
        self.max = max
        self.step = step
        # init base vectors
        self.basis = []
        base_min = self.hdc.random_hypervector(N)
        base_max = self.hdc.random_hypervector(N)
        # generate base vectors for each level
        num_intervals = int((self.max - self.min) // self.step)
        if (self.max - self.min) - num_intervals * self.step > 0:
            num_intervals += 1
        num_bits = int(self.N // num_intervals)
        print(f"num_intervals: {num_intervals}")
        print(f"num_bits: {num_bits}")
        for i in range(1, num_intervals):
            level = np.concatenate([base_min[:N - i*num_bits], base_max[N - i*num_bits:]])
            self.basis.append(level)
        # add underflow and overflow
        self.basis.insert(0, base_min)
        self.basis.append(base_max)
        
    
    def encode(self, val) -> np.ndarray:
        # find base vector
        idx = int((val - self.min) // self.step)
        # guard
        if idx < 0:
            idx = 0
        if idx >= len(self.basis):
            idx = len(self.basis) - 1
        encoded = self.basis[idx]
        return encoded
