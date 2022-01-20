from typing import List, Union
from enum import IntEnum

from datasets.data_readers.packet_readers.PacketReader import PacketReader, CSVWrapper


class UNSWDataset(IntEnum):
    HTTP = 0
    FTP = 1
    SMTP = 2
    SSH = 3
    DNS = 4
    FTP_DATA = 5
    IRC = 6


unsw_protocol_aliases = {
    "http": UNSWDataset.HTTP,
    "htp": UNSWDataset.FTP,
    "smtp": UNSWDataset.SMTP,
    "ssh": UNSWDataset.SSH,
    "dns": UNSWDataset.DNS,
    "ftp-data": UNSWDataset.FTP_DATA,
    "irc": UNSWDataset.IRC,
}


class UNSWPacketReader(PacketReader):
    def __init__(self,
                 packet_sources: Union[str, List[str], CSVWrapper],
                 is_mirai=False,
                 start=None,
                 end=None,
                 max_frame_count=None):
        super(UNSWPacketReader, self).__init__(packet_sources=packet_sources, start=start, end=end,
                                               max_frame_count=max_frame_count)

    def process_line(self, frame: List[str]):
        return frame
