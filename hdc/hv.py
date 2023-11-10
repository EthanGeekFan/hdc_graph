import numpy as np
from abc import ABC, abstractmethod

class HDCRepresentation(ABC):
    
    @classmethod
    @abstractmethod
    def random_hypervector(cls, N) -> np.ndarray:
        '''
        return a random hypervector of this representation dimension N
        '''
        pass
    
    @classmethod
    @abstractmethod
    def bundle(cls, els) -> np.ndarray:
        '''
        bundle the list of hypervectors
        '''
        pass
    
    @classmethod
    @abstractmethod
    def bind(cls, els) -> np.ndarray:
        '''
        bind the list of hypervectors
        '''
        pass
    
    @classmethod
    @abstractmethod
    def dist(cls, a, b) -> float:
        '''
        calculate distance between two hypervectors
        '''
        pass
    
    @classmethod
    @abstractmethod
    def permute(cls, hv, n) -> np.ndarray:
        '''
        permute the hypervector by n elements
        '''
        pass
    
    @classmethod
    @abstractmethod
    def sequence(cls, lst) -> np.ndarray:
        '''
        permute and bundle the list of hypervectors as a sequence
        '''
        pass
    
    @classmethod
    @abstractmethod
    def normalize(cls, hv) -> np.ndarray:
        '''
        normalize the hypervector to the representation range
        '''
        pass
