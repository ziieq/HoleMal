import time
import json
import socket

class Logger():
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, ip_list, ad_res_list):
        """
        ['192.168.10.14', '192.168.10.25', '192.168.10.3', '192.168.10.12']
        [1 1 0 1]
        [10  5  8]
        """
        time_local = time.localtime(time.time())
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
        res_dict = {'time': dt, 'results': []}
        ip_list = [socket.inet_ntoa(x) for x in ip_list if type(x) == bytes]
        res_list = []
        for i, ip in enumerate(ip_list):
            res_list.append({'ip': ip, 'AnomalyDetectorResult': str(ad_res_list[i])})

        res_dict['results'] = res_list
        res_json = json.dumps(res_dict)

        with open(self.log_path, 'a') as f:
            f.write(res_json+'\n')
