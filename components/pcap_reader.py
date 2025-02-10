import dpkt
import socket
import random


class PcapReader:
    def __init__(self, monitor_area):
        self.pcap = None
        self.f = None
        self.tmp_path = None
        self.monitor_area = {}
        for ip_segment in monitor_area:  # (ip, seg_pos)
            self.monitor_area[socket.inet_aton(ip_segment[0])[:ip_segment[1]]] = ip_segment

    def init(self, pcap_path):
        self.tmp_path = pcap_path
        self.f = open(pcap_path, 'rb')
        self.pcap = dpkt.pcap.Reader(self.f)

    def read_ip_layer_packet_from_pcap(self):
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
            for i in range(1, 5):
                if ip.src[:i] in self.monitor_area:
                    break
            else:
                continue

            # Random skip, for robust experiments
            # t = 0.8
            # if random.random() < t:
            #     continue

            yield ts, ip

    def read_all_packets_from_pcap(self):
        res = []
        while True:
            try:
                res.append(next(self.read_ip_layer_packet_from_pcap()))
            except Exception as e:
                break
        return res

    def read_chunk_from_pcap(self, chunk_size):
        res = []
        idx = 0
        while idx < chunk_size:
            try:
                res.append(next(self.read_ip_layer_packet_from_pcap()))
                idx += 1
            except Exception as e:
                break
        return res

    def read_chunk_from_memory(self, packets, chunk_size):
        chunk = []
        for ts, pkt in packets:
            chunk.append([ts, pkt])
            if len(chunk) >= chunk_size:
                yield chunk
                chunk.clear()
        if len(chunk) > 0:
            yield chunk

    def reset(self):
        self.pcap = None
        self.f.close()
        self.init(self.tmp_path)

