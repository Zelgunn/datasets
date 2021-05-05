from datasets.loaders.DatasetLoader import DatasetLoader, SingleSetConfig
from datasets.loaders.tfrecords.TFRecordSubsetLoader import TFRecordSubsetLoader


class TFRecordDatasetLoader(DatasetLoader):
    def __init__(self, config: SingleSetConfig):
        self.config = config
        super(TFRecordDatasetLoader, self).__init__()

    def init_subsets(self):
        for subset_name in self.config.subsets:
            self.subsets[subset_name] = TFRecordSubsetLoader(self.config, subset_name)
