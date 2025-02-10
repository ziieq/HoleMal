from line_profiler import LineProfiler
from functools import wraps
from components import logger
import joblib
from components.pcap_reader import PcapReader
from components.metadata_collector import MetadataCollector
from components.feature_extractor import FeatureExtractor
import tracemalloc


def func_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs)
        lp.print_stats()
        return
    return decorator

@func_line_time
def start(gen, ad, logger):
    mc = MetadataCollector()
    fe = FeatureExtractor()

    while True:
        try:
            chunk = next(gen)
        except Exception as e:
            break
        time_duration, meta_data_dict = mc.collect_metadata_by_chunk_from_memory(chunk)
        if not meta_data_dict:
            break
        ip_list, samples = fe.extract_features_v1(time_duration, meta_data_dict)

        ad_res = ad.predict(samples)
        logger.log(ip_list, ad_res)


if __name__ == '__main__':
    ad = joblib.load('pkl/ad_iot-23-gate.pkl')
    logger = logger.Logger('log/log.json')
    pcap_file = r'./components/pcap_to_test_components.pcap'
    # Note: Memory monitor will significantly slow down the speed. Turn it off when performing efficiency tests.
    is_monitor_memory = False

    monitor_area = [('192.168.0.0', 2), ('10.0.0.0', 1), ('172.16.0.0', 2)]
    reader = PcapReader(monitor_area)
    reader.init(pcap_file)
    packets = reader.read_chunk_from_pcap(100000)
    print('Read {} packets.'.format(len(packets)))
    gen = reader.read_chunk_from_memory(packets, 10000)

    if is_monitor_memory:
        tracemalloc.start()
        start(gen, ad, logger)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
        tracemalloc.stop()
    else:
        start(gen, ad, logger)