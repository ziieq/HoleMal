from HoleMal.components.pcap_reader import PcapReader


class MetadataCollector:
    def __init__(self):
        self.reader = None
        self.gen = None

    def init_reader(self, pcap_path, monitor_area=[('192.168.0.0', 2)]):
        self.reader = PcapReader(monitor_area=monitor_area)
        self.reader.init(pcap_path)
        self.gen = self.reader.read_ip_layer_packet_from_pcap()

    def collect_metadata_by_chunk_from_memory(self, chunk):
        meta_data_dict = {}
        tmp_time_dict = {}
        for (timestamp, pkt_ip) in chunk:

            # matrix, host_pair_dict
            meta_data_dict.setdefault(pkt_ip.src, [])

            # ts
            tmp_time_dict.setdefault(pkt_ip.src, timestamp)

            meta_data_dict[pkt_ip.src].append((pkt_ip.data.sport, pkt_ip.dst, pkt_ip.data.dport,
                                                  int(timestamp - tmp_time_dict[pkt_ip.src]), int(pkt_ip.len)))

            tmp_time_dict[pkt_ip.src] = timestamp

        # return time_duration, metadata_dict
        return chunk[-1][0] - chunk[0][0], meta_data_dict

    def collect_metadata_by_chunk_per_packet_from_pcap(self, chunk_size=10000):
        """
        This method reads 1 packet from local pcap files at a time and then extract the metadata of it.
        It will loop until a chunk_size of metadata is collected.
        """
        metadata_dict = {}
        tmp_time_dict = {}  # 用来计算时间间隙

        start_ts, end_ts = -1, -1
        for idx in range(chunk_size):
            try:
                timestamp, pkt_ip = next(self.gen)
                if start_ts == -1: start_ts = timestamp
            except Exception as e:
                break

            src_ip, sport, dst_ip, dport, pkt_len =\
                pkt_ip.src, pkt_ip.data.sport, pkt_ip.dst, pkt_ip.data.dport, int(pkt_ip.len)
            metadata_dict.setdefault(src_ip, [])
            tmp_time_dict.setdefault(src_ip, timestamp)
            metadata_dict[src_ip].append((sport, dst_ip, dport, timestamp - tmp_time_dict[src_ip], pkt_len))
            tmp_time_dict[src_ip] = timestamp
            end_ts = timestamp

        return end_ts-start_ts, metadata_dict

    def collect_metadata_by_chunk_from_pcap(self, chunk_size=10000):
        """
        This method reads chunk_size packets from local pcap files at a time and then extract the metadata of it.
        """
        metadata_dict = {}
        tmp_time_dict = {}  # 用来计算时间间隙
        pkts = self.reader.read_chunk_from_pcap(chunk_size)
        start_ts, end_ts = -1, -1
        for ts, pkt_ip in pkts:
            if start_ts == -1: start_ts = ts
            src_ip, sport, dst_ip, dport, pkt_len =\
                pkt_ip.src, pkt_ip.data.sport, pkt_ip.dst, pkt_ip.data.dport, int(pkt_ip.len)
            metadata_dict.setdefault(src_ip, [])
            tmp_time_dict.setdefault(src_ip, ts)
            metadata_dict[src_ip].append((sport, dst_ip, dport, ts - tmp_time_dict[src_ip], pkt_len))
            tmp_time_dict[src_ip] = ts
            end_ts = ts

        return end_ts-start_ts, metadata_dict


if __name__ == '__main__':
    monitor_area = [('192.168.0.0', 2), ('10.0.0.0', 1), ('172.16.0.0', 2)]
    pcap_file_path = './pcap_to_test_components.pcap'
    mc = MetadataCollector()
    mc.init_reader(pcap_file_path, monitor_area)
    res1 = mc.collect_metadata_by_chunk_per_packet_from_pcap()
    res2 = mc.collect_metadata_by_chunk_from_pcap()
    print(res1)
    print(res2)