import numpy as np
from line_profiler import LineProfiler
from functools import wraps


def get_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs)
        res_dict = lp.print_stats()
        return res_dict
    return decorator


def func_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs)
        res = lp.print_stats()
    return decorator


class FeatureExtractor:

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


    @get_line_time
    def extract_features(self, time_duration, meta_data_dict):
        samples = []
        host_features = {}
        for key in meta_data_dict.keys():

            meta_metrix = np.array(meta_data_dict[key])
            # Connection Behavior
            ## sport
            host_features['sport_uniq_ratio'] = len(set(list(meta_metrix[:, 0].astype(int)))) / len(list(meta_metrix[:, 0].astype(int)))
            host_features['sport_freq_mean'] = self.mean(self.get_freq_list(list(meta_metrix[:, 0].astype(int))))
            host_features['sport_freq_var'] = self.var(self.get_freq_list(list(meta_metrix[:, 0].astype(int))), host_features['sport_freq_mean'])
            host_features['sport_freq_ske'] = self.skewness(self.get_freq_list(list(meta_metrix[:, 0].astype(int))), host_features['sport_freq_mean'], host_features['sport_freq_var'])
            host_features['sport_freq_kur'] = self.kurtosis(self.get_freq_list(list(meta_metrix[:, 0].astype(int))), host_features['sport_freq_mean'], host_features['sport_freq_var'])

            ## ips
            host_features['ip_uniq_ratio'] = len(set(list(meta_metrix[:, 1]))) / len(list(meta_metrix[:, 1]))
            host_features['ip_freq_mean'] = self.mean(self.get_freq_list(list(meta_metrix[:, 1])))
            host_features['ip_freq_var'] = self.var(self.get_freq_list(list(meta_metrix[:, 1])), host_features['ip_freq_mean'])
            host_features['ip_freq_ske'] = self.skewness(self.get_freq_list(list(meta_metrix[:, 1])), host_features['ip_freq_mean'], host_features['ip_freq_var'])
            host_features['ip_freq_kur'] = self.kurtosis(self.get_freq_list(list(meta_metrix[:, 1])), host_features['ip_freq_mean'], host_features['ip_freq_var'])

            ## dport
            host_features['dport_uniq_ratio'] = len(set(list(meta_metrix[:, 2].astype(int)))) / len(list(meta_metrix[:, 2].astype(int)))
            host_features['dport_freq_mean'] = self.mean(self.get_freq_list(list(meta_metrix[:, 2].astype(int))))
            host_features['dport_freq_var'] = self.var(self.get_freq_list(list(meta_metrix[:, 2].astype(int))), host_features['dport_freq_mean'])
            host_features['dport_freq_ske'] = self.skewness(self.get_freq_list(list(meta_metrix[:, 2].astype(int))), host_features['dport_freq_mean'], host_features['dport_freq_var'])
            host_features['dport_freq_kur'] = self.kurtosis(self.get_freq_list(list(meta_metrix[:, 2].astype(int))), host_features['dport_freq_mean'], host_features['dport_freq_var'])

            # Network Vitality
            ## time
            host_features['time_interval_mean'] = self.mean(list(meta_metrix[:, 3].astype(float)))
            host_features['time_interval_var'] = self.var(list(meta_metrix[:, 3].astype(float)), host_features['time_interval_mean'])
            host_features['time_interval_ske'] = self.skewness(list(meta_metrix[:, 3].astype(float)), host_features['time_interval_mean'], host_features['time_interval_mean'])
            host_features['time_interval_kur'] = self.kurtosis(list(meta_metrix[:, 3].astype(float)), host_features['time_interval_mean'], host_features['time_interval_mean'])

            ## bytes
            host_features['bytes_mean'] = self.mean(list(meta_metrix[:, 4].astype(int)))
            host_features['bytes_var'] = self.var(list(meta_metrix[:, 4].astype(int)), host_features['bytes_mean'])
            host_features['bytes_ske'] = self.skewness(list(meta_metrix[:, 4].astype(int)), host_features['bytes_mean'], host_features['bytes_var'])
            host_features['bytes_kur'] = self.kurtosis(list(meta_metrix[:, 4].astype(int)), host_features['bytes_mean'], host_features['bytes_var'])

            ## host_features['speed']
            host_features['speed'] = len(list(meta_metrix[:, 4].astype(int)))/time_duration

            # Requested Service
            def is_service(service_list, dports):
                for x in service_list:
                    if x in dports:
                        return 1
                return 0

            host_features['ssh'] = is_service([22], list(meta_metrix[:, 2].astype(int)))
            host_features['http'] = is_service([80, 8080, 8081], list(meta_metrix[:, 2].astype(int)))
            host_features['https'] = is_service([443], list(meta_metrix[:, 2].astype(int)))
            host_features['telnet'] = is_service([23], list(meta_metrix[:, 2].astype(int)))
            host_features['mail'] = is_service([25, 50], list(meta_metrix[:, 2].astype(int)))
            host_features['dns'] = is_service([53], list(meta_metrix[:, 2].astype(int)))
            host_features['ntp'] = is_service([123], list(meta_metrix[:, 2].astype(int)))
            host_features['mqtt'] = is_service([1883], list(meta_metrix[:, 2].astype(int)))
            host_features['upnp'] = is_service([1900], list(meta_metrix[:, 2].astype(int)))
            host_features['mysql'] = is_service([3306], list(meta_metrix[:, 2].astype(int)))
            host_features['irc'] = is_service([6667], list(meta_metrix[:, 2].astype(int)))
            host_features['bittorrent'] = is_service([6881, 6882, 6883, 6884, 6885, 6886, 6887, 6888, 6889], list(meta_metrix[:, 2].astype(int)))

            samples.append(host_features)
        return samples

   





