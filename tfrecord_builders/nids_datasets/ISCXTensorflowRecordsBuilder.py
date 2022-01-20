from typing import Optional

from modalities import ModalityCollection
from datasets.tfrecord_builders import TFRecordBuilder
from datasets.modality_builders.PacketBuilder import DEFAULT_PACKET_FREQUENCY
from datasets.data_readers.packet_readers.ISCXPacketReader import ISCXDataset, iscx_protocol_aliases


class ISCXTensorflowRecordsBuilder(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 modalities: ModalityCollection,
                 version: ISCXDataset = None,
                 verbose=1):
        super(ISCXTensorflowRecordsBuilder, self).__init__(dataset_path=dataset_path,
                                                           shard_duration=shard_duration,
                                                           video_frequency=None,
                                                           audio_frequency=None,
                                                           modalities=modalities,
                                                           labels_frequency=DEFAULT_PACKET_FREQUENCY,
                                                           verbose=verbose)

        if version is None:
            version = self.guess_protocol(dataset_path)
            if version is None:
                raise ValueError("Could not determine version from dataset_path {} and version is `None`.".
                                 format(dataset_path))

        self.version = version

    @staticmethod
    def guess_protocol(dataset_path: str) -> Optional[ISCXDataset]:
        for alias in iscx_protocol_aliases:
            if alias in dataset_path:
                return iscx_protocol_aliases[alias]
        return None
