import numpy as np
from numpy import random
from functools import reduce
from hdc.hv import HDCRepresentation


class BSC(HDCRepresentation):
    @classmethod
    def random_hypervector(cls, N) -> np.ndarray:
        return random.binomial(1, 0.5, N)
    
    @classmethod
    def bundle(cls, els) -> np.ndarray:
        if len(els) == 1:
            return els[0]
        n = len(els)
        threshold = n // 2
        els = np.array(els)
        sums = np.sum(els, axis=0)
        if n % 2 == 1:
            hv = np.where(sums > threshold, 1, 0)
            return hv
        else:
            # when in tie randomly choose one
            hv = np.where(sums == threshold, random.binomial(1, 0.5), np.where(sums > threshold, 1, 0))
            return hv
        
    @classmethod
    def bind(cls, els) -> np.ndarray:
        return reduce(lambda a, b: np.logical_xor(a, b), els)
    
    @classmethod
    def dist(cls, a, b) -> float:
        return np.sum(np.logical_xor(a, b)) / len(a)
    
    @classmethod
    def permute(cls, hv, n) -> np.ndarray:
        return np.roll(hv, n)
    
    @classmethod
    def sequence(cls, lst) -> np.ndarray:
        n = len(lst)
        res = []
        for i in range(n):
            hv = lst[i]
            rotation = 8 * (n - i - 1)
            res.append(cls.permute(hv, rotation))
        return cls.bundle(res)
    
    @classmethod
    def normalize(cls, hv) -> np.ndarray:
        '''
        hv > 0.5 output 1
        hv < 0.5 output 0
        hv = 0.5 output random
        '''
        res = np.where(hv > 0.5, 1, np.where(hv < 0.5, 0, random.binomial(1, 0.5)))
        return res


if __name__ == '__main__':
    a = BSC.random_hypervector(100)
    b = BSC.random_hypervector(100)
    print('a vs b: ')
    print(BSC.sim(a, b))
    print(BSC.dist(a, b))
    print('a vs a: ')
    print(BSC.sim(a, a))
    print(BSC.dist(a, a))
    print('b vs b: ')
    print(BSC.sim(b, b))
    print(BSC.dist(b, b))
