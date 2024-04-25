from components.metadata_collector import MetadataCollector
from components.feature_extractor import FeatureExtractor
from components.pcap_reader import PcapReader
import pandas as pd


if __name__ == '__main__':
    """
    This is an example of processing a pcap file to csv file by HoleMal.
    """
    pcap_file_path = './dataset/test.pcap'
    dst_csv_path = './dataset/test.csv'

    reader = PcapReader()
    reader.read(pcap_file_path)
    gen = reader.get_packet_chunk_from_memory(size=1000, step=1000)

    feature_extractor = FeatureExtractor(want_ip=True, ip_in_feature=True)
    metadata_collector = MetadataCollector()

    csv_header = True

    while True:
        try:
            chunk = next(gen)
        except Exception as e:
            print(e)
            break

        # get metadata
        time_duration, meta_data_dict = metadata_collector.collect_metadata_from_chunk(chunk, b'\xc0\xa8')

        # get features
        ip_list, samples_ad, samples_mfd = feature_extractor.extract_features_deploy(time_duration, meta_data_dict)

        # build dataframe and write it
        df1 = pd.DataFrame(ip_list, columns=['ip'])
        df2 = pd.DataFrame(samples_ad, columns=['ip_uniq_ratio', 'dport_uniq_ratio', 'bytes_mean', 'speed'])
        df3 = pd.DataFrame(samples_mfd, columns=['sport_uniq_ratio', 'bytes_mean', 'speed', 'f_ssh', 'f_http'])
        df = pd.concat([df1, df2, df3], axis=1)
        df.to_csv(dst_csv_path, header=csv_header, index=False, mode='a')
        csv_header = False