import dpkt


class PcapReader:
    def __init__(self):
        self.pcap = None
        self.f = None
        self.tmp_path = None
        self.packets = None

    def read(self, pcap_path):
        self.tmp_path = pcap_path
        self.f = open(pcap_path, 'rb')
        self.pcap = dpkt.pcap.Reader(self.f)
        self.packets = self.get_packets()

    def get_ip_packet(self):
        for ts, pkt in self.pcap:
            eth = dpkt.ethernet.Ethernet(pkt)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            if isinstance(eth.data.data, dpkt.tcp.TCP):
                pass
            elif isinstance(eth.data.data, dpkt.udp.UDP):
                pass
            else:
                continue

            ip = eth.data
            yield ts, ip

    def get_eth_packet(self):
      for ts, pkt in self.pcap:
          eth = dpkt.ethernet.Ethernet(pkt)
          yield ts, eth

    def get_packets(self):
        res = []
        while True:
            try:
                res.append(next(self.get_ip_packet()))
            except Exception as e:
                break
        return res

    def reset(self):
        self.pcap = None
        self.f.close()
        self.read(self.tmp_path)

    def get_packet_chunk_from_memory(self, step=1000, size=1000):
        queue = []
        pkt_cnt = 0

        for ts, pkt_ip in self.packets:

            # 包是否符合条件
            if not isinstance(pkt_ip, dpkt.ip.IP):
                continue
            if not isinstance(pkt_ip.data, dpkt.tcp.TCP) and not isinstance(pkt_ip.data, dpkt.udp.UDP):
                continue

            # 加入队列
            queue.append((ts, pkt_ip))
            pkt_cnt += 1

            # 构成packet_chunk，则yield
            if pkt_cnt < size:
                continue
            yield queue

            # 清空前step
            del queue[:step]
            pkt_cnt = len(queue)