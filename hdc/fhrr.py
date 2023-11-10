import numpy as np
from numpy import random
from functools import reduce
from hdc.hv import HDCRepresentation


class FHRR(HDCRepresentation):
    @classmethod
    def random_hypervector(cls, N) -> np.ndarray:
        return random.uniform(-np.pi, np.pi, N)
    
    @classmethod
    def bundle(cls, els) -> np.ndarray:
        if len(els) == 1:
            return els[0]
        # add all hypervectors as unit complex exponentials
        els = np.array(els)
        els = np.exp(1j * els)
        # sum all unit complex exponentials
        hv = np.sum(els, axis=0)
        # normalize
        hv = np.angle(hv)
        return hv
    
    @classmethod
    def bind(cls, els) -> np.ndarray:
        # sum and normalize back to -pi to pi
        hv = np.sum(np.array(els), axis=0)
        # normalize back to -pi to pi
        hv = np.mod(hv + np.pi, 2 * np.pi) - np.pi
        return hv
    
    @classmethod
    def dist(cls, a, b) -> float:
        return 1 / (1 + np.mean(np.cos(a - b)))
    
    @classmethod
    def sim(cls, a, b) -> float:
        return np.mean(np.cos(a - b))
    
    @classmethod
    def permute(cls, hv, n) -> np.ndarray:
        return np.roll(hv, n)
    
    @classmethod
    def sequence(cls, lst) -> np.ndarray:
        n = len(lst)
        res = []
        for i in range(n):
            hv = lst[i]
            rotation = n - i - 1
            res.append(cls.permute(hv, rotation))
        return cls.bundle(res)
