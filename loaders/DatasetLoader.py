from abc import abstractmethod, ABC
from typing import Dict

from datasets.loaders import SubsetLoader


class DatasetLoader(ABC):
    def __init__(self):
        self.subsets: Dict[str, SubsetLoader] = {}
        self.init_subsets()

    @abstractmethod
    def init_subsets(self):
        raise NotImplementedError("`init_subsets` must be implemented in subclasses.")

    @property
    def train_subset(self) -> SubsetLoader:
        return self.subsets["Train"]

    @property
    def test_subset(self) -> SubsetLoader:
        if self.subsets["Test"].sample_count > 0:
            return self.subsets["Test"]
        else:
            return self.subsets["Train"]
