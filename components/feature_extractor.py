import numpy as np


class FeatureExtractor:

    def __init__(self):
        self.service_table = {
            # http
            80: 0,
            8080: 0,
            8081: 0,
            # https
            443: 1,
            # ssh
            22: 2,
            # telnet
            23: 3,
            # mail
            25: 4,
            50: 4,
            # DNS
            53: 5,
            # NTP
            123: 6,
            # MQTT
            1883: 7,
            # UPnP
            1900: 8,
            # MySQL
            3306: 9,
            # IRC
            6667: 10,
            # Bittorrent
            6881: 11,
            6882: 11,
            6883: 11,
            6884: 11,
            6885: 11,
            6886: 11,
            6887: 11,
            6888: 11,
            6889: 11,
        }
        self.service_type_cnt = len(set(self.service_table.values()))

    def mean(self, target_list):
        return sum(target_list)/len(target_list)

    def var(self, target_list, mean):
        return sum([pow(x-mean, 2) for x in target_list])/len(target_list)

    def get_freq_list(self, target_list):
        return [target_list.count(x) for x in set(target_list)]

    def skewness(self, target_list, mean, var):
        if not var:
            return -1
        std = pow(var, 0.5)
        return sum([pow((x-mean)/std, 3) for x in target_list])/len(target_list)

    def kurtosis(self, target_list, mean, var):
        if not var:
            return -1
        std = pow(var, 0.5)
        return sum([pow((x-mean)/std, 4) for x in target_list])/len(target_list)-3


    def extract_features_all(self, time_duration, meta_data_dict):
        ip_list = []
        samples = []

        for key in meta_data_dict.keys():
            meta_matrix = np.array(meta_data_dict[key])

            # Connection Behavior
            ## sport
            sports = list(meta_matrix[:, 0].astype(int))
            sport_uniq_ratio = len(set(sports)) / len(sports)
            cnt_list = self.get_freq_list(sports)
            sport_freq_mean = self.mean(cnt_list)
            sport_freq_var = self.var(cnt_list, sport_freq_mean)
            sport_freq_ske = self.skewness(cnt_list, sport_freq_mean, sport_freq_var)
            sport_freq_kur = self.kurtosis(cnt_list, sport_freq_mean, sport_freq_var)

            ## ips
            ips = list(meta_matrix[:, 1])
            ip_uniq_ratio = len(set(ips)) / len(ips)
            cnt_list = self.get_freq_list(ips)
            ip_freq_mean = self.mean(cnt_list)
            ip_freq_var = self.var(cnt_list, ip_freq_mean)
            ip_freq_ske = self.skewness(cnt_list, ip_freq_mean, ip_freq_var)
            ip_freq_kur = self.kurtosis(cnt_list, ip_freq_mean, ip_freq_var)

            ## dport
            dports = list(meta_matrix[:, 2].astype(int))
            dport_uniq_ratio = len(set(dports)) / len(dports)
            cnt_list = self.get_freq_list(dports)
            dport_freq_mean = self.mean(cnt_list)
            dport_freq_var = self.var(cnt_list, dport_freq_mean)
            dport_freq_ske = self.skewness(cnt_list, dport_freq_mean, dport_freq_var)
            dport_freq_kur = self.kurtosis(cnt_list, dport_freq_mean, dport_freq_var)

            # Network Vitality
            ## time
            ts_list = list(meta_matrix[:, 3].astype(float))
            time_mean = self.mean(ts_list)
            time_var = self.var(ts_list, time_mean)
            time_ske = self.skewness(ts_list, time_mean, time_var)
            time_kur = self.kurtosis(ts_list, time_mean, time_var)

            ## bytes
            bytes_list = list(meta_matrix[:, 4].astype(int))
            bytes_mean = self.mean(bytes_list)
            bytes_var = self.var(bytes_list, bytes_mean)
            bytes_ske = self.skewness(bytes_list, bytes_mean, bytes_var)
            bytes_kur = self.kurtosis(bytes_list, bytes_mean, bytes_var)

            ## Speed
            pkt_cnt = len(bytes_list)
            speed = pkt_cnt/time_duration

            # Requested Service
            service_feature = [0] * self.service_type_cnt
            range_list = [0, 0, 0]
            for dport in dports:
                ## service_feature
                if dport in self.service_table:
                    service_feature[self.service_table[dport]] = 1

            features = [key, sport_uniq_ratio, sport_freq_mean, sport_freq_var, sport_freq_ske, sport_freq_kur,
                        ip_uniq_ratio, ip_freq_mean, ip_freq_var, ip_freq_ske, ip_freq_kur,
                        dport_uniq_ratio, dport_freq_mean, dport_freq_var, dport_freq_ske, dport_freq_kur,
                        time_mean, time_var, time_ske, time_kur,
                        bytes_mean, bytes_var, bytes_ske, bytes_kur,
                        speed,
                        ] + service_feature

            samples.append(features)
        return samples

    def extract_features_v3(self, time_duration, meta_data_dict):
        samples = []
        ip_list = []
        for key in meta_data_dict.keys():
            meta_matrix = np.array(meta_data_dict[key])
            ips = list(meta_matrix[:, 1])
            ip_uniq_ratio = len(set(ips)) / len(ips)
            cnt_list = self.get_freq_list(ips)
            ip_freq_mean = self.mean(cnt_list)
            ip_freq_var = self.var(cnt_list, ip_freq_mean)

            ts_list = list(meta_matrix[:, 3].astype(float))
            time_mean = self.mean(ts_list)
            time_var = self.var(ts_list, time_mean)
            time_kur = self.kurtosis(ts_list, time_mean, time_var)
            bytes_list = list(meta_matrix[:, 4].astype(int))
            pkt_cnt = len(bytes_list)
            speed = pkt_cnt / time_duration
            samples.append([ip_uniq_ratio, ip_freq_var, time_kur, speed])
            ip_list.append(key)
        return ip_list, samples

    def extract_features_v2(self, time_duration, meta_data_dict):
        samples = []
        ip_list = []
        for key in meta_data_dict.keys():
            meta_matrix = np.array(meta_data_dict[key])
            sports = list(meta_matrix[:, 0].astype(int))
            sport_uniq_ratio = len(set(sports)) / len(sports)
            dports = list(meta_matrix[:, 2].astype(int))
            cnt_list = self.get_freq_list(dports)
            dport_freq_mean = self.mean(cnt_list)
            bytes_list = list(meta_matrix[:, 4].astype(int))
            bytes_mean = self.mean(bytes_list)
            pkt_cnt = len(bytes_list)
            speed = pkt_cnt / time_duration
            samples.append([sport_uniq_ratio, dport_freq_mean, bytes_mean, speed])
            ip_list.append(key)
        return ip_list, samples

    def extract_features_v1(self, time_duration, meta_data_dict):
        samples = []
        ip_list = []
        for key in meta_data_dict.keys():
            meta_matrix = np.array(meta_data_dict[key])
            bytes_list = list(meta_matrix[:, 4].astype(int))
            bytes_mean = self.mean(bytes_list)
            pkt_cnt = len(bytes_list)
            speed = pkt_cnt / time_duration
            samples.append([bytes_mean, speed])
            ip_list.append(key)
        return ip_list, samples


if __name__ == '__main__':
    from metadata_collector import MetadataCollector
    fe = FeatureExtractor()
    monitor_area = [('192.168.0.0', 2), ('10.0.0.0', 1), ('172.16.0.0', 2)]
    mc = MetadataCollector()
    mc.init_reader(r"./pcap_to_test_components.pcap", monitor_area=monitor_area)

    while True:
        time_duration, meta_data_dict = mc.collect_metadata_by_chunk_per_packet_from_pcap(chunk_size=10000)
        if not meta_data_dict:
            break
        res = fe.extract_features_all(time_duration, meta_data_dict)
        print(res)