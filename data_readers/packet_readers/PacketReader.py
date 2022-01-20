import numpy as np
from abc import abstractmethod
from typing import List, Union

from datasets.utils.CSVWrapper import CSVWrapper


class PacketReader(object):
    """
    :param packet_sources:
        A string (filepath of a csv file) or a list of strings.
    :param start:
        (Optional) The start (in number of packets) of the stream. Used to only read a sub-part.
    :param end:
        (Optional) The end (in number of packets) of the stream. Used to only read a sub-part.
    """

    def __init__(self,
                 packet_sources: Union[str, List[str], CSVWrapper],
                 start=None,
                 end=None,
                 max_frame_count=None,
                 ):

        if isinstance(packet_sources, (str, list)):
            packet_sources = CSVWrapper(packet_sources)

        self.csv_wrapper = packet_sources
        self.packet_size = self.csv_wrapper.column_count

        # region End / Start
        if max_frame_count is None:
            max_frame_count = self.csv_wrapper.get_row_count()

        if end is None:
            self.end = max_frame_count
        elif end < 0:
            self.end = max_frame_count + end
        else:
            self.end = min(end, max_frame_count)

        if start is None:
            self.start = 0
        elif start < 0:
            self.start = max_frame_count + start
        elif start < max_frame_count:
            self.start = start
        else:
            raise ValueError("`start` is after the end of the stream "
                             "{}(start) >= {}(frame_count)".format(start, max_frame_count))

        if self.end <= self.start:
            raise ValueError("`start` must be less than `end`, got {}(start) and {}(end)".format(self.start, self.end))
        # endregion

    def __iter__(self):
        for i, frame in enumerate(self.csv_wrapper):
            if (i >= self.start) and self.keep_line(i, frame):
                frame = self.process_line(frame)
                yield frame

            if (i + 1) == self.end:
                break

        self.csv_wrapper.close()

    def read_all(self, stack=True):
        packets = [packet for packet in self]
        if stack:
            packets = np.stack(packets, axis=0)
        return packets

    @abstractmethod
    def keep_line(self, index: int, frame: List[str]) -> bool:
        raise NotImplementedError("You must implement this method in sub-classes.")

    @abstractmethod
    def process_line(self, frame: List[str]):
        raise NotImplementedError("You must implement this method in sub-classes.")

    @property
    def frame_count(self) -> int:
        return self.end - self.start
