import dpkt
import pcap     # pcap-ct
from scapy.all import *


class Listener(object):
    def __init__(self, if_name=None):
        self.sniffer = pcap.pcap(if_name, promisc=True, immediate=False, timeout_ms=50)

    def get_ip_packet_with_tran(self):
        for ts, pkt in self.sniffer:
            eth = dpkt.ethernet.Ethernet(pkt)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            if isinstance(eth.data.data, dpkt.tcp.TCP):
                pass
            elif isinstance(eth.data.data, dpkt.udp.UDP):
                pass
            else:
                continue
            yield ts, eth.data

    def get_eth_packet(self):
        for ts, pkt in self.sniffer:
            eth = dpkt.ethernet.Ethernet(pkt)
            yield ts, eth

    def get_packet_chunk(self, step=500, size=1000):
        queue = []
        pkt_cnt = 0
        for ts, pkt in self.sniffer:
            # 包是否符合条件
            eth = dpkt.ethernet.Ethernet(pkt)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            if not isinstance(eth.data.data, dpkt.tcp.TCP) and not isinstance(eth.data.data, dpkt.udp.UDP):
                continue

            # 加入队列
            queue.append((ts, eth.data))
            pkt_cnt += 1

            # 构成packet_chunk，则yield
            if pkt_cnt < size:
                continue

            yield queue

            # 清空前step
            del queue[:step]
            pkt_cnt = len(queue)


if __name__ == '__main__':
    listener = Listener()
    while True:
        ts, ip = next(listener.get_ip_packet_with_tran())
        print(ip)

