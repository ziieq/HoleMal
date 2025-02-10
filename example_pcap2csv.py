from components.metadata_collector import MetadataCollector
from components.feature_extractor import FeatureExtractor
import pandas as pd
import os


if __name__ == '__main__':
    """
    This is an example of processing a pcap file to csv file by HoleMal.
    """
    pcap_file_path = './components/pcap_to_test_components.pcap'
    dst_csv_path = './example.csv'
    if os.path.exists(dst_csv_path):
        os.remove(dst_csv_path)

    monitor_area = [('192.168.0.0', 2), ('10.0.0.0', 1), ('172.16.0.0', 2)]
    mc = MetadataCollector()
    mc.init_reader(pcap_file_path, monitor_area)
    fe = FeatureExtractor()

    while True:
        time_duration, meta_data_dict = mc.collect_metadata_by_chunk_per_packet_from_pcap(chunk_size=10000)
        if not meta_data_dict:
            break
        # Get all host-level features (You need to run cost-sensitive feature selector before deployment to identify the feature subset.)
        samples = fe.extract_features_all(time_duration, meta_data_dict)
        print('Extract 1 chunk. {} hosts detected.'.format(len(samples)))
        df = pd.DataFrame(samples, columns=['ip', 'sport_uniq_ratio', 'sport_freq_mean', 'sport_freq_var', 'sport_freq_ske', 'sport_freq_kur',
                            'ip_uniq_ratio', 'ip_freq_mean', 'ip_freq_var', 'ip_freq_ske', 'ip_freq_kur',
                            'dport_uniq_ratio', 'dport_freq_mean', 'dport_freq_var', 'dport_freq_ske', 'dport_freq_kur',
                            'time_interval_mean', 'time_interval_var', 'time_interval_ske', 'time_interval_kur',
                            'bytes_mean', 'bytes_var', 'bytes_ske', 'bytes_kur',
                            'speed',
                            'http', 'https', 'ssh', 'telnet', 'mail', 'dns', 'ntp', 'mqtt', 'upnp', 'mysql', 'irc',
                            'bittorrent'])

        df.to_csv(dst_csv_path, header=(not os.path.exists(dst_csv_path)), index=False, mode='a')

    print('* Processing is complete.')
