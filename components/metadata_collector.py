from line_profiler import LineProfiler
from functools import wraps


def func_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs)
        lp.print_stats()
        return
    return decorator


class MetadataCollector(object):
    def __init__(self):
        pass

    def collect_metadata_from_chunk(self, chunk, monitor_area):
        meta_data_dict = {}
        tmp_time_dict = {}
        for (timestamp, pkt_ip) in chunk:

            if pkt_ip.src[:2] == monitor_area:

                # matrix, host_pair_dict
                meta_data_dict.setdefault(pkt_ip.src, ([], {}))

                # ts
                tmp_time_dict.setdefault(pkt_ip.src, timestamp)

                meta_data_dict[pkt_ip.src][0].append((pkt_ip.data.sport, pkt_ip.dst, pkt_ip.data.dport,
                                                      int(timestamp - tmp_time_dict[pkt_ip.src]), int(pkt_ip.len)))

                tmp_time_dict[pkt_ip.src] = timestamp

        # return time_duration, metadata_dict
        return chunk[-1][0] - chunk[0][0], meta_data_dict


"""
INPUT: PacketSet(Traffic of hosts behind the gateway by a fixed number) , MonitoringRange
OUTPUT: MetadataDict(The dictionary of metadata)
Initialize MetadataDict 
for Packet in PacketSet do
    ip = ExtractSourceIP(Packet)
    if ip in MonitoringRange then
        Metadata = Extract_Metadata(Packet)
        Append(MetadataDict[ip], Metadata)
    end if
end for
"""