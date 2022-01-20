from typing import List, Union
from enum import IntEnum

from datasets.data_readers.packet_readers.PacketReader import PacketReader, CSVWrapper


class ISCXDataset(IntEnum):
    SSH = 0
    POP = 1
    ICMP = 2
    SMTP = 3
    HTTPWeb = 4
    HTTPImageTransfer = 5
    FTP = 6
    DNS = 7
    IMAP = 8


iscx_protocol_aliases = {
    "SSH": ISCXDataset.SSH,
    "POP": ISCXDataset.POP,
    "ICMP": ISCXDataset.ICMP,
    "SMTP": ISCXDataset.SMTP,
    "HTTPWeb": ISCXDataset.HTTPWeb,
    "HTTPImageTransfer": ISCXDataset.HTTPImageTransfer,
    "FTP": ISCXDataset.FTP,
    "DNS": ISCXDataset.DNS,
    "IMAP": ISCXDataset.IMAP,
}


class ISCXPacketReader(PacketReader):
    def __init__(self,
                 packet_sources: Union[str, List[str], CSVWrapper],
                 start=None,
                 end=None,
                 max_frame_count=None):
        super(ISCXPacketReader, self).__init__(packet_sources=packet_sources, start=start, end=end,
                                               max_frame_count=max_frame_count)

    def process_line(self, frame: List[str]):
        return frame
