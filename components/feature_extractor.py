import numpy as np

class FeatureExtractor(object):

    def __init__(self, want_ip=True, ip_in_feature=False):
        self.want_ip = want_ip
        self.ip_in_feature = ip_in_feature

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

    def extract_features(self, time_duration, meta_data_dict):
        ip_list = []
        samples = []

        for key in meta_data_dict.keys():

            meta_metrix = np.array(meta_data_dict[key][0])

            # Connection Behavior
            ## sport
            sports = list(meta_metrix[:, 0].astype(int))
            sport_uniq_ratio = len(set(sports)) / len(sports)
            cnt_list = self.get_freq_list(sports)
            sport_freq_mean = self.mean(cnt_list)
            sport_freq_var = self.var(cnt_list, sport_freq_mean)
            sport_freq_ske = self.skewness(cnt_list, sport_freq_mean, sport_freq_var)
            sport_freq_kur = self.kurtosis(cnt_list, sport_freq_mean, sport_freq_var)

            ## ips
            ips = list(meta_metrix[:, 1])
            ip_uniq_ratio = len(set(ips)) / len(ips)
            cnt_list = self.get_freq_list(ips)
            ip_freq_mean = self.mean(cnt_list)
            ip_freq_var = self.var(cnt_list, ip_freq_mean)
            ip_freq_ske = self.skewness(cnt_list, ip_freq_mean, ip_freq_var)
            ip_freq_kur = self.kurtosis(cnt_list, ip_freq_mean, ip_freq_var)

            ## dport
            dports = list(meta_metrix[:, 2].astype(int))
            dport_uniq_ratio = len(set(dports)) / len(dports)
            cnt_list = self.get_freq_list(dports)
            dport_freq_mean = self.mean(cnt_list)
            dport_freq_var = self.var(cnt_list, dport_freq_mean)
            dport_freq_ske = self.skewness(cnt_list, dport_freq_mean, dport_freq_var)
            dport_freq_kur = self.kurtosis(cnt_list, dport_freq_mean, dport_freq_var)

            # Network Vitality
            ## time
            ts_list = list(meta_metrix[:, 3].astype(int))
            time_mean = self.mean(ts_list)
            time_var = self.var(ts_list, time_mean)
            time_ske = self.skewness(ts_list, time_mean, time_var)
            time_kur = self.kurtosis(ts_list, time_mean, time_var)

            ## bytes
            bytes_list = list(meta_metrix[:, 4].astype(int))
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
                ## range_list
                if dport <= 1023:
                    range_list[0] = 1
                elif dport <= 49151:
                    range_list[1] = 1
                else:
                    range_list[2] = 1
            # range_list = [x/len(dports) for x in range_list]

            features = [sport_uniq_ratio, sport_freq_mean, sport_freq_var, sport_freq_ske, sport_freq_kur,
                        ip_uniq_ratio, ip_freq_mean, ip_freq_var, ip_freq_ske, ip_freq_kur,
                        dport_uniq_ratio, dport_freq_mean, dport_freq_var, dport_freq_ske, dport_freq_kur,
                        time_mean, time_var, time_ske, time_kur,
                        bytes_mean, bytes_var, bytes_ske, bytes_kur,
                        speed,
                        ] + service_feature + range_list

            if self.want_ip:
                if not self.ip_in_feature:
                    ip_list.append(key)
                else:
                    features.insert(0, key)

            samples.append(features)
        return ip_list, samples

    def extract_features_deploy(self, time_duration, meta_data_dict):
        ip_list = []
        samples_ad = []
        samples_mfd = []

        for key in meta_data_dict.keys():

            meta_metrix = np.array(meta_data_dict[key][0])

            # Connection Behavior
            # ip
            ips = meta_metrix[:, 1]
            ip_uniq_ratio = len(set(ips)) / len(ips)
            # dport
            dports = meta_metrix[:, 2]
            dport_uniq_ratio = len(set(dports)) / len(dports)
            # sport
            sports = meta_metrix[:, 0]
            sport_uniq_ratio = len(set(sports)) / len(sports)

            # Network Vitality
            # bytes
            bytes_list = list(meta_metrix[:, 4].astype(int))
            bytes_mean = self.mean(bytes_list)

            speed = len(bytes_list)/time_duration

            # Requested Service
            f_ssh = 0
            f_http = 0
            tmp_dict = {}
            for dport in dports:
                tmp_dict[dport] = 1

            if 22 in tmp_dict:
                f_ssh = 1
            if 80 in tmp_dict or 8080 in tmp_dict or 8081 in tmp_dict:
                f_http = 1

            features_ad = (ip_uniq_ratio,  dport_uniq_ratio, bytes_mean, speed)
            features_mfd = (sport_uniq_ratio, bytes_mean, speed, f_ssh, f_http)
            if self.want_ip:
                ip_list.append(key)

            samples_ad.append(features_ad)
            samples_mfd.append(features_mfd)
        return ip_list, samples_ad, samples_mfd






