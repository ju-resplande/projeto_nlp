from abc import ABC, abstractmethod
from typing import List


def AbsModel(ABC):
    @abstractmethod
    def get_logits(self, X: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embbed(self, X: List[str]) -> List[List[float]]:
        pass