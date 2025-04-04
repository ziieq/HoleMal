import shutil
import time
from components.metadata_collector import MetadataCollector
from components.feature_extractor import FeatureExtractor
import pandas as pd
import os
from multiprocessing import Pool, Lock
import multiprocessing


def editcap(editcap_path, split_size, src_pcap_path, dst_pcap_path):
    cmd = fr'{editcap_path} -F pcap -c {split_size} {src_pcap_path} {dst_pcap_path}'
    print(cmd)
    os.system(cmd)


def HoleMal_process(pcap_file_path, monitor_area):
    mc = MetadataCollector()
    fe = FeatureExtractor()
    mc.init_reader(pcap_file_path, monitor_area)
    samples_list = []
    while True:
        time_duration, meta_data_dict = mc.collect_metadata_by_chunk_per_packet_from_pcap(chunk_size=10000)
        if not meta_data_dict:
            break
        # Get all host-level features (You need to run cost-sensitive feature selector before deployment to identify the feature subset.)
        samples = fe.extract_features_all(time_duration, meta_data_dict)
        samples_list.append(samples)
    return samples_list


def print_error(value):
    print("error: ", value)


def output(samples_list):
    lock.acquire()
    for samples in samples_list:
        df = pd.DataFrame(samples, columns=['ip', 'sport_uniq_ratio', 'sport_freq_mean', 'sport_freq_var', 'sport_freq_ske',
                                            'sport_freq_kur',
                                            'ip_uniq_ratio', 'ip_freq_mean', 'ip_freq_var', 'ip_freq_ske', 'ip_freq_kur',
                                            'dport_uniq_ratio', 'dport_freq_mean', 'dport_freq_var', 'dport_freq_ske',
                                            'dport_freq_kur',
                                            'time_interval_mean', 'time_interval_var', 'time_interval_ske',
                                            'time_interval_kur',
                                            'bytes_mean', 'bytes_var', 'bytes_ske', 'bytes_kur',
                                            'speed',
                                            'http', 'https', 'ssh', 'telnet', 'mail', 'dns', 'ntp', 'mqtt', 'upnp', 'mysql',
                                            'irc',
                                            'bittorrent'])
        print(f'{time.time()}: Extract 1 tmp pcap. {len(samples)} hosts detected.')
        df.to_csv(dst_csv_path, header=(not os.path.exists(dst_csv_path)), index=False, mode='a')
    lock.release()


lock = Lock()
if __name__ == '__main__':
    """
    This is an example of processing a pcap file to csv file by HoleMal.
    """

    pcap_file_path = './components/pcap_to_test_components.pcap'
    dst_csv_path = './example.csv'
    editcap_path = r'E:\EngineeringWare\NetworkTraffic\Wireshark\editcap.exe'
    tmp_dir = './tmp'
    worker_num = 10
    split_size = 200000

    multiprocessing.log_to_stderr()
    if os.path.exists(dst_csv_path):
        os.remove(dst_csv_path)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    pool = Pool(worker_num)

    monitor_area = [('192.168.0.0', 2), ('10.0.0.0', 1), ('172.16.0.0', 2)]

    start_time, end_time = time.time(), -1

    editcap(editcap_path, split_size, pcap_file_path, os.path.join(tmp_dir, 'tmp.pcap'))

    for root, dirs, files in os.walk(tmp_dir):
        for file in files:
            if not file.endswith('pcap'):
                continue
            tmp_pcap_file = os.path.join(root, file)
            pool.apply_async(func=HoleMal_process, args=(tmp_pcap_file, monitor_area), callback=output, error_callback=print_error)
    pool.close()
    pool.join()

    print('* Processing is complete.')
    end_time = time.time()
    print(f'* Time consumption: {end_time-start_time}')