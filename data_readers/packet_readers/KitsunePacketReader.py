import numpy as np
from typing import List, Union, Iterator

from datasets.data_readers.packet_readers.PacketReader import PacketReader, CSVWrapper


class KitsunePacketReader(PacketReader):
    def __init__(self,
                 packet_sources: Union[str, List[str], CSVWrapper],
                 is_mirai=False,
                 start=None,
                 end=None,
                 max_frame_count=None):
        super(KitsunePacketReader, self).__init__(packet_sources=packet_sources, start=start, end=end,
                                                  max_frame_count=max_frame_count)
        self.is_mirai = is_mirai
        if self.is_mirai:
            self.packet_size -= 1

    def process_line(self, frame: List[str]) -> Iterator[np.ndarray]:
        if self.is_mirai:
            frame = frame[1:]
        frame = np.asarray(frame).astype(np.float32)
        return frame

    def keep_line(self, index: int, frame: List[str]) -> bool:
        return True
