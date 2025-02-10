import tqdm
from HoleMal.components.feature_extractor_for_FCTT import FeatureExtractor
from HoleMal.components.metadata_collector import MetadataCollector
import pandas as pd
import os
import math
import dpkt
import sys


class Redirect:
    """
    将cmd输出捕获到变量里，获取line_profiler的输出。
    """
    def __init__(self):
        self.backup = sys.stdout
        self.content = ''

    def write(self, s):
        self.content += s

    def transform(self):
        sys.stdout = self

    def restore(self):
        sys.stdout = self.backup


class FeatureTimeDictProcessor:
    def __init__(self):
        pass

    def get_feature_time_dict_from_pcap(self, pcap_path):
        print(pcap_path)
        monitor_area = [('192.168.0.0', 2), ('10.0.0.0', 1), ('172.16.0.0', 2)]
        mc = MetadataCollector()
        mc.init_reader(pcap_path, monitor_area)
        fe = FeatureExtractor(want_ip=True, ip_in_feature=False)

        feature_time_dict = {}
        loop_cnt = 0
        while True:

            time_duration, meta_data_dict = mc.collect_metadata_by_chunk_per_packet_from_pcap(chunk_size=10000)
            if not meta_data_dict:
                break

            redirect = Redirect()
            redirect.transform()
            fe.extract_features(time_duration, meta_data_dict)
            redirect.restore()
            print(redirect.content)
            line_time_list = redirect.content.split('\n')
            loop_cnt += 1
            for line in line_time_list:
                if 'host_features[' in line:
                    feature_name = line.split("'")[1]
                    words_list = line.split()
                    try:
                        feature_time_dict.setdefault(feature_name, 0)
                        feature_time_dict[feature_name] += float(words_list[2])
                    except Exception as e:
                        feature_time_dict[feature_name] = 0
        for key in feature_time_dict.keys():
            feature_time_dict[key] /= loop_cnt
        return feature_time_dict

    def get_feature_time_dict_from_all_pcaps(self, dir_path='./pcap'):

        # 对每个pcap得到一个检测结果字典
        feature_time_dict_list = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if not file.endswith('pcap'):
                    continue
                src_file_path = os.path.join(root, file)
                feature_time_dict_list.append(self.get_feature_time_dict_from_pcap(src_file_path))

        # 所有结果取平均值
        final_feature_time_dict = {}
        for key in feature_time_dict_list[0].keys():
            time_total = 0
            for feature_time_dict in feature_time_dict_list:
                time_total += feature_time_dict[key]
            time_average = time_total / len(feature_time_dict_list)
            final_feature_time_dict[key] = time_average
        return final_feature_time_dict

    def write_time_dict_2_csv(self, feature_time_dict, dst_csv_path):
        if not feature_time_dict: return
        df_feature_time = pd.DataFrame.from_dict(feature_time_dict, orient='index', columns=['time'])
        df_feature_time.to_csv(dst_csv_path, index=True)

    def get_pcap_duration(self, pcap_path):
        # res = os.popen(" capinfos -u -T {}".format(pcap_path)).read()
        # dur = float(res.split('\n')[1].split('	')[-1].strip())
        with open(pcap_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            start, end = None, None
            # 遍历数据包
            for timestamp, buf in pcap:
                start = timestamp if start is None else start
                end = timestamp
            dur = end - start
        return dur

    def get_flow_frequency(self, pcap_dir_path, pcap_session_dir_path):
        """
        以流频率为基础计算权重
        :param pcap_dir_path: 原始6个场景PCAP文件
        :param pcap_session_dir_path: 切分后的6个场景PCAP文件
        :return: 频率字典
        """
        freq_dict = {}

        # 获取每个场景的时长
        for root, dirs, files in os.walk(pcap_dir_path):
            for file in files:
                if file.split('.')[-1] != 'pcap':
                    continue
                file_path = os.path.join(root, file)
                duration = self.get_pcap_duration(file_path)
                scenario = os.path.basename(os.path.dirname(file_path))
                freq_dict[scenario] = duration

        # 计算每个场景的权重
        for root, dirs, files in os.walk(pcap_session_dir_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                cnt = len(os.listdir(dir_path))
                freq_dict[dir] = cnt / freq_dict[dir]
                print(dir, cnt, freq_dict[dir])

        weight_dict = {}
        for key in freq_dict.keys():
            weight_dict[key] = pow(math.log(1 + freq_dict[key], 10), 0.5)
            #weight_dict[key] = math.log(1 + pow(freq_dict[key], 0.5), 10)

        # 归一化
        max_v = max(weight_dict.values())
        for key in weight_dict.keys():
            weight_dict[key] = weight_dict[key] / max_v

        return weight_dict

    def get_feature_time_by_weighted_average(self, dict_dir_path, weight_dict):
        df_dict = {}
        for root, dirs, files in os.walk(dict_dir_path):
            for file in files:

                if file.split('.')[-1] != 'csv':
                    continue

                file_path = os.path.join(root, file)

                scenario = file.split('_')[-1].strip('.csv')

                df_dict[scenario] = pd.read_csv(file_path, header=0, index_col=0)
                weight = weight_dict[scenario]
                df_dict[scenario] = weight * df_dict[scenario]

        df_total = None
        for key in df_dict.keys():
            if df_total is None:
                df_total = df_dict[key]
                continue
            df_total += df_dict[key]
        df_total /= sum(weight_dict.values())
        return df_total

    def normalize(self, df):
        tmp_list = df['time'].to_list()
        t_max, t_min = max(tmp_list), 0
        for index, row in df.iterrows():
            df.loc[index] = (df.loc[index] - t_min) / (t_max - t_min)
        return df


def dict2csv(src_dict_path, dst_csv_path):
    """
    unit:
        windows: 1e-7
        linux on raspberry pi: 1e-9
    """
    unit = 1e-7
    with open(src_dict_path, 'r') as f:
        d = eval(f.read())

    for key in d.keys():
        d[key] = [d[key] / unit]
    df = pd.DataFrame.from_dict(d, orient='index')
    print(df.head())
    df.to_csv(dst_csv_path, header=False)


if __name__ == '__main__':
    ftdp = FeatureTimeDictProcessor()
    time_dict = ftdp.get_feature_time_dict_from_all_pcaps(r'../pcap')
    with open('./time_dict.txt', 'w') as f:
        f.write(str(time_dict))