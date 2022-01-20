import numpy as np
import os
from typing import List, Union, Optional
from enum import IntEnum

from datasets.utils.CSVWrapper import CSVWrapper
from datasets.data_readers.packet_readers.PacketReader import PacketReader


class CIDDSSubset(IntEnum):
    ExternalServer = 0
    OpenStack = 1


cidds_subset_names = {
    CIDDSSubset.ExternalServer: "ExternalServer",
    CIDDSSubset.OpenStack: "OpenStack",
}


class CIDDSNetworkProtocol(IntEnum):
    ICMP = 0
    UDP = 1
    TCP = 2
    GRE = 3
    IGMP = 4


cidds_protocol_aliases = {protocol.name: protocol for protocol in CIDDSNetworkProtocol}


def guess_network_protocol(string: str) -> Optional[CIDDSNetworkProtocol]:
    for protocol in CIDDSNetworkProtocol:
        if protocol.name in string:
            return protocol
    return None


class CIDDSFields(IntEnum):
    DURATION = 1
    PROTOCOL = 2
    SOURCE_IP = 3
    SOURCE_PORT = 4
    DESTINATION_IP = 5
    DESTINATION_PORT = 6
    PACKETS = 7
    BYTES = 8
    FLOWS = 9
    FLAGS = 10
    TOS = 11
    LABEL = 12


def process_ip(ip_address: str) -> np.ndarray:
    """
    Parses an ip address and returns a vector with 4 values with dtype float32.

        :param ip_address: A string, either formatted as an IPv4 address, or "DNS", or "EXT_SERVER", or "OPENSTACK",
            "ATTACKER" or two values separated with an underscore.
        :return: A vector containing 4 floating point values.
    """
    if ip_address == "DNS":
        return np.full(shape=[4], fill_value=-1e3, dtype=np.float32)
    elif '.' in ip_address:
        return np.asarray(ip_address.split('.'), dtype=np.float32)
    elif ("EXT_SERVER" in ip_address) or ("OPENSTACK" in ip_address) or ("ATTACKER" in ip_address):
        return np.zeros(shape=[4], dtype=np.float32)
    else:
        return np.asarray([0.0, 0.0, *ip_address.split('_')], dtype=np.float32)


def process_bytes_count(bytes_count: str) -> np.ndarray:
    if 'M' in bytes_count:
        tmp = bytes_count.split(' ')
        bytes_count = None
        for element in tmp:
            if element != '':
                bytes_count = element
                break
        bytes_count = np.float32(bytes_count) * 1e6
    else:
        bytes_count = np.float32(bytes_count)
    return bytes_count


def process_flags(flags: str) -> np.ndarray:
    if 'x' in flags:
        tmp = int(flags, 16)
        tmp = [tmp >> i & 1 for i in range(7, -1, -1)]
        return np.float32(tmp)
    else:
        if len(flags) != 6:
            raise ValueError("Expected flags to contain 6 symbols, got {}.".format(len(flags)))
        tmp = [0.0 if symbol == "." else 1.0 for symbol in flags]
        tmp = np.float32([0.0, 0.0] + tmp)
        return tmp


def process_protocol(protocol: str) -> CIDDSNetworkProtocol:
    return cidds_protocol_aliases[protocol.strip()]


def process_label(packet_class: str) -> bool:
    return packet_class != "normal"


# duration; processIP(source); source port?; processIP(destination); destination port?; packets?;
class CIDDSPacketReader(PacketReader):
    def __init__(self,
                 packet_sources: Union[str, List[str], CSVWrapper],
                 start=None,
                 end=None,
                 max_frame_count=None):
        super(CIDDSPacketReader, self).__init__(packet_sources=packet_sources, start=start, end=end,
                                                max_frame_count=max_frame_count)

    def keep_line(self, index: int, frame: List[str]) -> bool:
        return frame[CIDDSFields.DURATION] != "Duration"

    def process_line(self, frame: List[str]):
        duration = np.float32(frame[CIDDSFields.DURATION])
        source_ip = process_ip(frame[CIDDSFields.SOURCE_IP])
        source_port = np.float32(frame[CIDDSFields.SOURCE_PORT])
        destination_ip = process_ip(frame[CIDDSFields.DESTINATION_IP])
        destination_port = np.float32(frame[CIDDSFields.DESTINATION_PORT])
        packets = np.float32(frame[CIDDSFields.PACKETS])
        bytes_count = process_bytes_count(frame[CIDDSFields.BYTES])
        flows = np.float32(frame[CIDDSFields.FLOWS])
        flags = process_flags(frame[CIDDSFields.FLAGS])
        tos = np.float32(frame[CIDDSFields.TOS])

        features = [duration, source_ip, source_port, destination_ip, destination_port, packets, bytes_count, flows,
                    flags, tos]
        features_array = np.empty(shape=[sum(feature.size for feature in features)], dtype=np.float32)
        i = 0
        for feature in features:
            features_array[i: i + feature.size] = feature
            i += feature.size
        features = features_array

        protocol = process_protocol(frame[CIDDSFields.PROTOCOL])
        label = process_label(frame[CIDDSFields.LABEL])

        return protocol, features, label

    def read_all(self, stack=True):
        protocols, samples, labels = [], [], []
        for protocol, features, label in self:
            protocols.append(protocol)
            samples.append(features)
            labels.append(label)

        if stack:
            protocols = np.stack(protocols, axis=0)
            samples = np.stack(samples, axis=0)
            labels = np.stack(labels, axis=0)

        return protocols, samples, labels

    def read_all_and_export(self, target_folder: str):
        protocols, features, labels = self.read_all(stack=True)

        np.save(os.path.join(target_folder, "protocols.npy"), protocols)
        np.save(os.path.join(target_folder, "features.npy"), features)
        np.save(os.path.join(target_folder, "labels.npy"), labels)
