from line_profiler import LineProfiler
from functools import wraps
from components import feature_extractor
from components import metadata_collector, listener, logger
import joblib

def func_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs)
        lp.print_stats()
        return
    return decorator


def main_listen(listener, metedata_collector, feature_extractor, ad, mfd, logger):
    print('HoleMal starts listening...')
    isMFD = False

    while True:
        chunk = next(listener.get_packet_chunk())

        # 队满，采集元数据
        time_duration, metadata_dict = metedata_collector.collect_metadata_from_chunk(chunk, b'\xc0\xa8')

        # 元数据提取多个特征样本
        ip_list, samples_ad, samples_mfd = feature_extractor.extract_features_deploy(time_duration, metadata_dict)

        # 预测
        ad_res = ad.predict(samples_ad)

        # MFD
        mfd_res = None
        if isMFD:
            if 1 in ad_res:
                indexes = [i for i in range(len(ad_res)) if ad_res[i] == 1]
                f = [samples_mfd[i] for i in indexes]
                mfd_res = mfd.predict(f)
        logger.log(ip_list, ad_res, mfd_res)


@func_line_time
def main_read(gen, metedata_collector, feature_extractor, ad, mfd, logger):

    metadata_dict_cnt = 0
    ad_samples_cnt = 0
    mfd_samples_cnt = 0

    # tracemalloc.start()
    while True:
        try:
            chunk = next(gen)
        except Exception:
            break

        # 队满，采集元数据
        time_duration, metadata_dict = metedata_collector.collect_metadata_from_chunk(chunk, b'\xc0\xa8')
        metadata_dict_cnt += 1
        # 元数据提取多个特征样本
        ip_list, samples_ad, samples_mfd = feature_extractor.extract_features_deploy(time_duration, metadata_dict)
        ad_samples_cnt += len(samples_ad)
        # print(ad_samples_cnt)
        # 预测
        ad_res = ad.predict(samples_ad)
        # MFD
        mfd_res = None
        if isMFD:
            if 1 in ad_res:
                indexs = [i for i in range(len(ad_res)) if ad_res[i] == 1]
                f = [samples_mfd[i] for i in indexs]
                mfd_samples_cnt += len(f)
                mfd_res = mfd.predict(f)

                # print(mfd_res)
        logger.log(ip_list, ad_res, mfd_res)

        # current, peak = tracemalloc.get_traced_memory()
        # print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    print('metadata_dict: {}'.format(metadata_dict_cnt))
    print('ad sample: {}'.format(ad_samples_cnt))
    print('mfd sample: {}'.format(mfd_samples_cnt))


if __name__ == '__main__':
    listener_or_read = 0
    isMFD = True
    ad = joblib.load('pkl/ad.pkl')
    mfd = joblib.load('pkl/mfd.pkl')
    logger = logger.Logger('log/log.json')

    host_metadata_collector = metadata_collector.MetadataCollector()
    feature_extractor = feature_extractor.FeatureExtractor()

    if listener_or_read == 0:
        # start to listen from NIC
        listener = listener.Listener(None)
        main_listen(listener, host_metadata_collector, feature_extractor, ad, mfd, logger)
    else:
        # read local pcap
        from components.pcap_reader import PcapReader
        reader = PcapReader()
        reader.read(r"./dataset/test.pcap")
        print(len(reader.packets))
        gen = reader.get_packet_chunk_from_memory(size=1000, step=1000)
        main_read(gen, host_metadata_collector, feature_extractor, ad, mfd, logger)
