from typing import Type
from hdc.hv import HDCRepresentation
import numpy as np

class ItemMem:
    def __init__(self, hdc: Type[HDCRepresentation]) -> None:
        self.hdc = hdc
        self.mem = {}
        self.caches = {}
        
    def cache(self, key, hv):
        key = int(key)
        if key not in self.caches:
            self.caches[key] = []
        self.caches[key].append(hv)
        
    def build(self):
        for key, hvs in self.caches.items():
            self.mem[key] = self.hdc.bundle(hvs)
        self.caches = {}
        
    def query(self, hv):
        min_dist = float("inf")
        res = None
        for key, mem_hv in self.mem.items():
            dist = self.hdc.dist(hv, mem_hv)
            if dist < min_dist:
                min_dist = dist
                res = key
        return res
    
    def profile(self, hv = None):
        if hv is None:
            # profile the distance between all pairs of items
            keys = [int(k) for k in self.mem.keys()]
            res = [[-1 for _ in range(len(keys))] for _ in range(len(keys))]
            for i in keys:
                for j in keys:
                    dist = self.hdc.dist(self.mem[i], self.mem[j])
                    if i == j:
                        print(f"dist {i} vs {j}: {dist}")
                    res[i][j] = dist
                    res[j][i] = dist
            return res
        else:
            # profile the distance between hv and all items
            keys = list(self.mem.keys())
            res = [-1 for _ in range(len(keys))]
            for i in range(len(keys)):
                dist = self.hdc.dist(hv, self.mem[keys[i]])
                res[i] = dist
            return res

class HighResItemMem(ItemMem):
    def __init__(self, hdc: type[HDCRepresentation], lr = 1) -> None:
        super().__init__(hdc)
        self.lr = lr
    
    def cache(self, key, hv):
        key = int(key)
        if key not in self.caches:
            self.caches[key] = [np.zeros(len(hv)), 0]
        if key not in self.mem:
            sim = 0
        else:
            sim = self.hdc.dist(self.mem[key], hv)
        self.caches[key][0] += hv * (1 - sim) * self.lr
        self.caches[key][1] +=(1 - sim) * self.lr
        
    def decache(self, key, hv):
        key = int(key)
        if key not in self.caches:
            self.caches[key] = [np.zeros(len(hv)), 0]
        if key not in self.mem:
            sim = 0
        else:
            sim = self.hdc.dist(self.mem[key], hv)
        self.caches[key][0] -= hv * (1 - sim) * self.lr
        self.caches[key][1] -= (1 - sim) * self.lr
    
    def build(self):
        for key, cache_line in self.caches.items():
            new_mem = self.hdc.normalize(cache_line[0] / cache_line[1])
            self.mem[key] = new_mem if cache_line[1] > 0 else np.logical_not(new_mem)