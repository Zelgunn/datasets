from typing import Dict

from datasets.loaders import SubsetLoader, SingleSetConfig


class DatasetLoader(object):
    def __init__(self, config: SingleSetConfig):
        self.config = config

        self.subsets: Dict[str, SubsetLoader] = {}
        for subset_name in config.subsets:
            self.subsets[subset_name] = SubsetLoader(config, subset_name)

    @property
    def train_subset(self) -> SubsetLoader:
        return self.subsets["Train"]

    @property
    def test_subset(self) -> SubsetLoader:
        if len(self.subsets["Test"].subset_folders) > 0:
            return self.subsets["Test"]
        else:
            return self.subsets["Train"]
