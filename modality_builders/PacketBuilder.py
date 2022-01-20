import numpy as np
from typing import List, Dict, Type

from modalities import Modality, ModalityCollection, NetworkPacket
from datasets.modality_builders import ModalityBuilder
from datasets.data_readers import PacketReader

DEFAULT_PACKET_FREQUENCY = 500


class PacketBuilder(ModalityBuilder):
    def __init__(self,
                 shard_duration: float,
                 modalities: ModalityCollection,
                 packet_reader: PacketReader,
                 packets_mins: np.ndarray,
                 packets_maxs: np.ndarray,
                 ):
        super(PacketBuilder, self).__init__(shard_duration=shard_duration,
                                            source_frequency=DEFAULT_PACKET_FREQUENCY,
                                            modalities=modalities)

        if not isinstance(packet_reader, PacketReader):
            raise ValueError("Parameter `packet_reader` must be a sub-class of PacketReader.")

        self.reader = packet_reader
        self.packets_mins = packets_mins
        self.packets_maxs = packets_maxs
        self.packets_ranges = (packets_maxs - packets_mins) if self.packets_mins is not None else None
        if np.any(self.packets_ranges == 0):
            self.packets_ranges = np.where(self.packets_ranges == 0, np.ones_like(packets_mins), self.packets_ranges)

    @classmethod
    def supported_modalities(cls):
        return [NetworkPacket]

    def check_shard(self, frames: np.ndarray) -> bool:
        return frames.shape[0] > 1

    def process_shard(self, frames: np.ndarray) -> Dict[Type[Modality], np.ndarray]:
        shard: Dict[Type[Modality], np.ndarray] = {}

        if NetworkPacket in self.modalities:
            network_packets = frames
            if self.packets_mins is not None:
                network_packets = (frames - self.packets_mins) / self.packets_ranges
            shard[NetworkPacket] = network_packets

        return shard

    def get_buffer_shape(self, frame: np.ndarray = None) -> List[int]:
        max_shard_size = self.get_source_max_shard_size()
        return [max_shard_size, self.reader.packet_size]

    @property
    def source_frame_count(self):
        return self.reader.frame_count

    def close(self):
        self.reader.close()
