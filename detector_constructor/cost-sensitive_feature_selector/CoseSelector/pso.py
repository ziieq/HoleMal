import os.path
from sklearn.model_selection import train_test_split
import numpy as np
import math
import random
import pandas as pd
import trainer_sklearn, model


class Particle:

    def __init__(self, dim, speed):
        self.pos = np.array([0] * dim)  # 粒子当前位置
        self.fitness = None
        self.speed = speed  # np.random.random(size=(dim))
        self.pbest_pos = self.pos  # 粒子历史最好位置
        self.pbest_fitness = math.inf


class PSO:
    def __init__(self, density=0.1, matrix_num=2, iter_n=20, object_func=None, dim=69):
        self.w = 1  # 惯性因子
        self.c1 = 0.5  # 自我认知学习因子
        self.c2 = 0.5  # 社会认知学习因子
        self.global_best_pos = None  # 种群当前最好位置
        self.global_best_fitness = - math.inf
        self.density = density  # 种群中粒子数量
        self.particle_num = round(1 / density)
        self.matrix_num = matrix_num
        self.particle_list = None  # 种群
        self.iter_n = iter_n  # 迭代次数
        self.object_func = object_func
        self.dim = dim

    # 初始化种群
    def PIOSV(self, density=0.8):

        def generate_vector(density, flag_list):
            # particle_num 向上取整后会多出
            if flag_list.count(1) == len(flag_list):
                flag_list = [0] * self.dim

            # 生成正交的稀疏向量
            flag_idx_list = []  # 保存可选的方向
            for i in range(len(flag_list)):
                if flag_list[i] == 0:
                    flag_idx_list.append(i)

            tmp_num = int(self.dim * density)
            if tmp_num < len(flag_idx_list):
                index_list = random.sample(flag_idx_list, tmp_num)
            else:
                index_list = flag_idx_list
            vector = np.zeros(shape=(self.dim))

            for i in index_list:
                vector[i] = random.uniform(0.3, 0.7)
                flag_list[i] = 1

            return vector

        particle_list = []

        for _ in range(self.matrix_num):
            flag_list = [0] * self.dim  # 保存已经被选的方向
            for i in range(self.particle_num):
                particle = Particle(self.dim, generate_vector(density, flag_list))
                particle.fitness = self.object_func(particle.pos)
                particle.pbest_pos = particle.pos
                particle_list.append(particle)

        # 找到种群中的最优位置
        self.set_global_best(particle_list)
        return particle_list

    def initPopulationRandom(self):
        particle_list = []
        for i in range(self.particle_num * self.matrix_num):
            vector = np.zeros(shape=(self.dim))
            for i in range(self.dim):
                vector[i] = random.uniform(0.0, 1.0)
            particle = Particle(self.dim, vector)
            particle.fitness = self.object_func(particle.pos)
            particle.pbest_pos = particle.pos
            particle_list.append(particle)

        self.set_global_best(particle_list)
        return particle_list

    def run(self):
        # 初始化种群
        print('\n *** INIT ***\n')
        self.particle_list = self.PIOSV()

        global_best_fitness_list = []
        # 迭代
        for i in range(self.iter_n):  # n_i
            print('\n *** 第{}次iter. ***\n'.format(i + 1))
            # 更新速度和位置
            print('\n *** UPDATE ***\n')
            self.update(self.particle_list)
            # 更新种群中最好位置
            print('\n *** G_BEST ***\n')
            self.set_global_best(self.particle_list)
            global_best_fitness_list.append(self.global_best_fitness)

        print(self.global_best_pos, self.global_best_fitness)
        import pandas as pd
        df = pd.DataFrame(global_best_fitness_list).T
        df.to_csv('./fit_line.csv', header=False, index=False, mode='a')
        return self.global_best_pos, self.global_best_fitness

    # 更新速度和位置
    def update(self, particle_list):
        for i in range(len(particle_list)):
            print('* 第{}个粒子.'.format(i + 1))
            particle = particle_list[i]
            # 速度更新
            # sign_matrix1 = np.array([random.choice((-1, 1)) for _ in range(self.dim)])
            # sign_matrix2 = np.array([random.choice((-1, 1)) for _ in range(self.dim)])

            sign_matrix1 = np.array([1 for _ in range(self.dim)])
            sign_matrix2 = np.array([1 for _ in range(self.dim)])

            speed = self.w * particle.speed + \
                    self.c1 * sign_matrix1 * np.random.random(size=(self.dim)) * (particle.pbest_pos - particle.pos) + \
                    self.c2 * sign_matrix2 * np.random.random(size=(self.dim)) * (self.global_best_pos - particle.pos)
            # 位置更新
            pos = particle.pos + speed
            for x in range(len(pos)):
                if pos[x] > 1:
                    pos[x] = 1
                elif pos[x] < 0:
                    pos[x] = 0
            particle.pos = pos
            particle.speed = speed
            # 更新适应度
            particle.fitness = self.object_func(particle.pos)
            pbest_fitness = particle.pbest_fitness
            # 是否需要更新本粒子历史最好位置
            if particle.fitness > pbest_fitness:
                particle.pbest_pos = particle.pos
                particle.pbest_fitness = particle.fitness

    # 找到全局最优解
    def set_global_best(self, particle_list):
        for particle in particle_list:
            if particle.fitness > self.global_best_fitness:
                self.global_best_pos = particle.pos
                self.global_best_fitness = particle.fitness


class RecoFitness:
    def __init__(self, train_set, eval_set, feature_time_dict, isTorch=0, isEval=0):

        self.isTorch = isTorch
        self.isEval = isEval  # 是否用于验证，用于验证则将各类loss分开返回

        # 读取特征计算时间字典、特征列表
        self.feature_time_dict = feature_time_dict
        self.feature_list_all = list(self.feature_time_dict.keys())

        # 读取数据集
        self.train_set = train_set
        self.evaluate_set = eval_set

    def evaluate_sklearn(self, feature_select_list):
        fs = model.RandomForest()
        acc, f1, precision, recall, auc, FPR = trainer_sklearn.evaluate_model(fs, self.train_set, self.evaluate_set, feature_select_list)
        return acc, f1, precision, recall, auc, FPR

    def get_feature_list(self, p_select_list):
        # 获取选择的特征列表
        # select_list = [pick_by_p(x) for x in p_select_list[0]]
        select_list = [1 if x > 0.5 else 0 for x in p_select_list[0]]
        feature_select_list = []
        for i in range(len(select_list)):
            if select_list[i] == 1:
                feature_select_list.append(self.feature_list_all[i])
        return feature_select_list

    def func(self, x, n):
        return pow(x, n)

    def get_f1_fitness(self, feature_select_list):
        eva_res = self.evaluate_sklearn(feature_select_list)
        bce_loss, f1 = 0, eva_res[1]
        f1_loss = f1 if f1 < 1 else 1

        return self.func(f1_loss, 4)

    def get_time_fitness(self, feature_select_list):
        x = sum([self.feature_time_dict[f] for f in feature_select_list])
        total = sum(list(self.feature_time_dict.values()))
        x = x/total
        return self.func(1-x, 2)  # x * x - 2 * x + 1

    def get_feature_num_fitness(self, feature_select_list):
        feature_total = len(list(self.feature_time_dict.keys()))

        # feature_num_fitness = (feature_total - len(feature_select_list) + 1) / feature_total
        feature_num_fitness = (len(feature_select_list) - feature_total) / (1 - feature_total)
        return self.func(feature_num_fitness, 0.5) if len(feature_select_list) != 0 else 0

    def pick_by_p(self, p):
        x = random.random()
        if x < p:
            return 1
        return 0

    def get_fitness(self, *p_select_list):
        # 获取已选择的特征列表
        feature_select_list = self.get_feature_list(p_select_list)
        if not feature_select_list:
            print('Feature_selecte_list is None.')
            return 0

        fitness = 4 * self.get_f1_fitness(feature_select_list) + 2 * self.get_time_fitness(feature_select_list) + self.get_feature_num_fitness(feature_select_list)

        print('选择特征：{}，总Fitness：{}'.format(feature_select_list, fitness))
        return fitness

def split_dataset(df):
    df = df.sample(frac=1)
    df_y = df['label']
    df_x = df.iloc[:, :-1].astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.4)
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    return train_set, test_set


if __name__ == '__main__':
    dataset_name = 'iot-23-gate'
    loop_cnt = 20
    chunk_size = 10000
    time_dict_path = '../time_extractor/time_dict_pi_1w.txt'    # Feature consumption time table (FCTT)
    res_path = 'res_pso.csv'

    for _ in range(0, loop_cnt):
        # Read data
        df = pd.read_csv('../../../dataset/chunk/{}/df_{}.csv'.format(dataset_name, chunk_size))
        df = df.iloc[:, 1:]
        train_set, eval_set = split_dataset(df)

        with open(time_dict_path, 'r') as f:
            time_dict = eval(f.read())
        feature_list_all = list(time_dict.keys())

        reco = RecoFitness(train_set, eval_set, time_dict)
        pso = PSO(object_func=reco.get_fitness, dim=len(list(time_dict.keys())), iter_n=20)
        # Start running
        gbest_x, gbest_fit = pso.run()
        feature_list = []
        for i in range(len(gbest_x)):
            if gbest_x[i] > 0.5:
                feature_list.append(feature_list_all[i])

        # Compute results
        cost = sum([time_dict[x] for x in feature_list])
        fs = model.RandomForest()
        acc, f1, precision, recall, auc, FPR = trainer_sklearn.evaluate_model(fs, train_set, eval_set, feature_list, is_print=False)

        res = {
            'dataset_name': [dataset_name],
            'feature_list': [feature_list],
            'fit': [gbest_fit],
            'cost': [cost],
            'acc': [acc],
            'f1': [f1],
            'precision': [precision],
            'recall': [recall],
            'auc': [auc],
            'FPR': [FPR],
        }

        df = pd.DataFrame(res)

        df.to_csv(res_path, mode='a', index=False, header=(not os.path.exists(res_path)))
