import gym
import numpy as np
import torch
from gym.utils import EzPickle
from Params import configs


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


class satellite_edge(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 maxtasks,
                 max_Men):
        EzPickle.__init__(self)
        self.maxtasks = maxtasks
        self.n_j = n_j
        self.maxMen = max_Men
        self.step_count = 0
        self.L = 50
        self.number_of_jobs = n_j
        self.number_of_tasks_on_satellite = 0
        self.busy_men_on_satellite = 0


    def reset(self, batch,data): #  初始化
        self.batch = batch
        self.job_finish_time_on_local = np.zeros(self.batch * self.maxtasks).reshape((self.batch, -1))
        self.job_finish_time_on_edge = np.zeros(self.batch * self.maxtasks).reshape((self.batch, -1))
        self.job_finish_time_on_satellite = np.zeros(self.batch * self.maxtasks).reshape((self.batch, -1))

        self.step_count = 0
        # print(self.step_count)

        # self.place = data[-1]

        self.dur_l = np.array(data[2], dtype=np.single)
        self.dur_e = np.array(data[3], dtype=np.single)
        self.dur_s = np.array(data[4], dtype=np.single)
        self.dur_se = np.array(data[5], dtype=np.single)
        self.dur_sa = np.array(data[6], dtype=np.single)
        self.datasize = np.array(data[0], dtype=np.single)
        self.T = np.array(data[1], dtype=np.single)
        # print('####',self.T.dtype)


        self.I = np.full(shape=(self.batch,self.n_j,3), fill_value=0, dtype=bool)
        self.LBs = np.zeros((self.batch,self.n_j,3), dtype=np.single)
        self.Fi = np.zeros((self.batch,self.n_j,3), dtype=np.single)
        self.LBm = np.zeros((self.batch, self.n_j, 1), dtype=np.single)
        self.Fim = np.zeros((self.batch, self.n_j, 1), dtype=np.single)

        self.place_time = np.zeros((self.batch, 3), dtype=np.single)
        self.task_mask = np.full(shape=self.T.shape, fill_value=0, dtype=bool)
        self.place_mask = np.full(shape=self.LBs.shape, fill_value=0, dtype=bool)
        # print('T',self.task_mask.shape)


        for i in range(self.batch):
            for j in range(self.n_j):
                self.LBs[i][j][0] = self.dur_l[i][j]
                self.LBs[i][j][1] = self.dur_se[i][j] + self.dur_e[i][j]
                self.LBs[i][j][2] = self.dur_s[i][j]+ self.dur_sa[i][j]

                self.Fi[i][j][0] = self.T[i][j] - self.LBs[i][j][0]
                self.Fi[i][j][1] = self.T[i][j] - self.LBs[i][j][1]
                self.Fi[i][j][2] = self.T[i][j] - self.LBs[i][j][2]

                self.LBm[i][j][0] = min(self.LBs[i][j][0],self.LBs[i][j][1],self.LBs[i][j][2])
                self.Fim[i][j][0] = self.Fi[i][j][2]

        task_feas = np.concatenate((self.LBm.reshape(self.batch, self.n_j, 1),
                                    self.Fim.reshape(self.batch, self.n_j, 1),
                                    self.task_mask.reshape(self.batch, self.n_j, 1), )
                                   , axis=2)

        # print(self.I[0])

        return task_feas,self.task_mask, self.place_time


    def step(self,task_action,p_action):#  根据智能体的动作更新特征
        for i in range(self.batch):
            if p_action[i] == 2:
                # Satellite execution
                earlist_ind = np.argmin(self.job_finish_time_on_satellite[i])
                self.job_finish_time_on_satellite[i][earlist_ind] = self.LBs[i][task_action[i]][2]
                min_ind = np.argmin(self.job_finish_time_on_satellite[i])
                self.place_time[i][2] = self.job_finish_time_on_satellite[i][min_ind]
            elif p_action[i] == 1:
                # Edge execution
                earlist_ind_edge = np.argmin(self.job_finish_time_on_edge[i])
                self.job_finish_time_on_edge[i][earlist_ind_edge] = self.LBs[i][task_action[i]][1]
                min_ind_edge = np.argmin(self.job_finish_time_on_edge[i])
                self.place_time[i][1] = self.job_finish_time_on_edge[i][min_ind_edge]


        reward = np.zeros((self.batch, 1))
        # print(self.job_finish_time_on_satellite[0])

        for i in range(self.batch):
            if self.LBs[i][task_action[i]][p_action[i]] <= self.T[i][task_action[i]]:
                reward[i] = self.LBs[i][task_action[i]][p_action[i]]
            else:
                reward[i] = self.LBs[i][task_action[i]][p_action[i]] * 10


        # print(p_action[0])
        # print('reward',reward[0])
        earlist_time = np.zeros((self.batch,3))
        for i in range(self.batch):
            earlist_time[i][0] = min(self.job_finish_time_on_local[i])
            earlist_time[i][1] = min(self.job_finish_time_on_edge[i])
            earlist_time[i][2] = min(self.job_finish_time_on_satellite[i])
        # print(earlist_time[0])
        # print(place_time[0])

        for i in range(self.batch):
            self.I[i][task_action[i]][0] = True
            self.I[i][task_action[i]][1] = True
            self.I[i][task_action[i]][2] = True

        for b in range(self.batch):
            self.task_mask[b][task_action[b]] = True

        for i in range(self.batch):
            for j in range(self.n_j):
                if self.I[i][j][0] == False and self.I[i][j][1] == False and self.I[i][j][2] == False:
                    # local
                    job_readytime_a_l = 0
                    compute_readytime_a_l = 0
                    job_startime_a_l = max(job_readytime_a_l, compute_readytime_a_l)
                    job_finishitime_a_l = job_startime_a_l + self.dur_l[i][j]
                    self.LBs[i][j][0] = job_finishitime_a_l
                    self.Fi[i][j][0] = self.T[i][j] - self.LBs[i][j][0]

                    # edge
                    job_readytime_a_e = self.dur_se[i][j]
                    compute_readytime_a_e = min(self.job_finish_time_on_edge[i])
                    job_startime_a_e = max(job_readytime_a_e, compute_readytime_a_e)
                    job_finishitime_a_e = job_startime_a_e + self.dur_e[i][j]
                    self.LBs[i][j][1] = job_finishitime_a_e
                    self.Fi[i][j][1] = self.T[i][j] - self.LBs[i][j][1]

                    #satellite
                    jobreadytime_a_s = self.dur_s[i][j]
                    compute_readytime_a_s = min(self.job_finish_time_on_satellite[i])
                    job_startime_a_s = max(jobreadytime_a_s, compute_readytime_a_s)
                    job_finishitime_a_s = job_startime_a_s + self.dur_sa[i][j]
                    self.LBs[i][j][2] = job_finishitime_a_s
                    self.Fi[i][j][2] = self.T[i][j] - self.LBs[i][j][2]
                    self.LBm[i][j][0] = min(self.LBs[i][j][0],self.LBs[i][j][1],self.LBs[i][j][2])
                    self.Fim[i][j][0] = self.Fi[i][j][2]


        task_feas = np.concatenate((self.LBm.reshape(self.batch, self.n_j, 1),
                                    self.Fim.reshape(self.batch, self.n_j, 1),
                                    self.task_mask.reshape(self.batch, self.n_j, 1),
                                    )
                                   , axis=2)

        # print('LBs',self.LBs[0])
        # print('F',self.Fi[0])


        # print(self.task_mask[0])
        return task_feas, self.task_mask, self.place_time,reward
