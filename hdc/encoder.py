from abc import ABC, abstractmethod
import numpy as np

class Encoder(ABC):
    @abstractmethod
    def encode(self, val) -> np.ndarray:
        pass
