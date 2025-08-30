from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEngine(ABC):
    """Abstract base class for all computational engines."""

    @abstractmethod
    def compute(self, *args, **kwargs) -> List[np.ndarray]:
        """The main computation method for the engine."""
        pass
