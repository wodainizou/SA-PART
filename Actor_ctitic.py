from Params import configs
from Saenv import satellite_edge
import torch.nn as nn
from Transformer import Encoder1,Encoder
import torch

from Agent_untils import select_action
from Agent_untils import greedy_select_action
from Agent_untils import sample_select_action

import torch.nn.functional as F
from torch import optim
import numpy as np

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


class task_actor(nn.Module):#  任务选择代理-输出动作和对应动作的概率

    def __init__(self,
                 batch,
                 hidden_dim,
                 M):
        super().__init__()
        self.M = M
        self.hidden_dim = hidden_dim
        self.task_encoder = Encoder1(Inputdim=configs.input_dim1,
                               embedding_size=configs.hidden_dim,
                               M=M).to(DEVICE)
        self.batch = batch

        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)


    def forward(self,data,index,feas,mask,action_pro,train):
        mask = torch.from_numpy(mask).to(DEVICE)
        tag = torch.LongTensor(self.batch).to(DEVICE)
        for i in range(self.batch):
            tag[i] = configs.n_j * i

        C = 10

        nodes, grapha = self.task_encoder(feas)
        torch.cuda.empty_cache()

        q = grapha

        dk = self.hidden_dim / self.M

        query = self.wq(q)  # (batch, embedding_size)
        query = torch.unsqueeze(query, dim=1)
        query = query.expand(self.batch, configs.n_j , self.hidden_dim)

        key = self.wk(nodes)

        temp = query * key
        temp = torch.sum(temp, dim=2)
        temp = temp / (dk ** 0.5)
        temp = torch.tanh(temp) * C
        temp.masked_fill_(mask, float('-inf'))
        p = F.softmax(temp, dim=1)
        ppp = p.view(1, -1).squeeze()
        p = torch.unsqueeze(p, dim=2)

        if train == 1:
            action_index = select_action(p)
        elif train == 2:
            action_index = sample_select_action(p)
        else:
            action_index = greedy_select_action(p)


        action_pro[tag + index] = ppp[tag + action_index]#

        dur_l = np.array(data[2], dtype=np.single)
        dur_e = np.array(data[3], dtype=np.single)
        dur_s = np.array(data[4], dtype=np.single)
        dur_se = np.array(data[5], dtype=np.single)
        dur_sa = np.array(data[6], dtype=np.single)
        datasize = np.array(data[0], dtype=np.single)
        T = np.array(data[1], dtype=np.single)
        process_time = np.zeros((self.batch, 3),dtype=np.single)

        for i in range(self.batch):
            process_time[i][0] = dur_l[i][action_index[i]]
            process_time[i][1] = dur_e[i][action_index[i]]
            process_time[i][2] = dur_sa[i][action_index[i]]
        return action_index,action_pro,process_time   ##(batch,1)

class place_actor(nn.Module):# 选择计算节点-输出动作和对应于该动作的概率
    def __init__(self,
                 batch,
                 hidden_dim,
                 M,
                ):
        super().__init__()
        self.M = M
        self.hidden_dim = hidden_dim
        self.p_encoder = Encoder1(Inputdim=configs.input_dim2,
                                         embedding_size=hidden_dim,
                                         M=M).to(DEVICE)

        self.batch = batch

        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)

        self.w = nn.Linear(hidden_dim, hidden_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, index, task_op, p_action_pro, place_time, process_time, train):
        p_feas = np.concatenate((place_time.reshape(self.batch, 3, 1),
                                 process_time.reshape(self.batch, 3, 1))
                                , axis=2)

        C = 10

        p_tag = torch.LongTensor(self.batch).to(DEVICE)

        for i in range(self.batch):
            p_tag[i] = configs.n_j * i
        p_tag1 = torch.LongTensor(self.batch).to(DEVICE)
        for i in range(self.batch):
            p_tag1[i] = 2 * i

        nodes, grapha = self.p_encoder(p_feas)

        torch.cuda.empty_cache()

        q = grapha
        dk = self.hidden_dim / self.M

        query = self.wq(q)  # (batch, embedding_size)
        query = torch.unsqueeze(query, dim=1)
        query = query.expand(self.batch, 3, self.hidden_dim)

        key = self.wk(nodes)

        temp = query * key
        temp = torch.sum(temp, dim=2)
        temp = temp / (dk ** 0.5)
        temp = torch.tanh(temp) * C
        p = F.softmax(temp, dim=1)
        ppp = p.view(1, -1).squeeze()
        p = torch.unsqueeze(p, dim=2)

        if train == 1:
            action_index = select_action(p)
        elif train == 2:
            action_index = sample_select_action(p)
        else:
            action_index = greedy_select_action(p)

        p_action_pro[p_tag + index] = ppp[p_tag1 + action_index]
        return action_index, p_action_pro

class actor_critic(nn.Module): #两个代理协同工作
    def __init__(self,
                batch,
                 hidden_dim,
                 M,
                 device):

        super().__init__()
        self.M = M
        self.hidden_dim = hidden_dim

        self.env = satellite_edge(n_j=configs.n_j,
                              maxtasks=configs.maxtask,
                              max_Men=configs.Men)

        self.actor1 = task_actor(batch=batch,
                                     hidden_dim=hidden_dim,
                                     M=M)#  任务选择代理

        self.actor2 = place_actor(batch=batch,
                                 hidden_dim=hidden_dim,
                                 M=M) #  计算节点选择代理

        self.batch = batch

    def forward(self,data,train):
        action_pro = torch.zeros(self.batch, configs.n_j).to(DEVICE).view(1, -1).squeeze().to(DEVICE)
        p_action_pro = torch.zeros(self.batch, configs.n_j).to(DEVICE).view(1, -1).squeeze().to(DEVICE)
        task_feas,task_mask,place_time = self.env.reset(self.batch, data)
        task_seq_list = []
        p_op_list = []
        rewards = 0
        q = torch.zeros((self.batch,1)).to(DEVICE)

        for i in range(configs.n_j):
            index = i
            task_op,action_pro,process_time = self.actor1(data,index,task_feas, task_mask,action_pro, train)  #  选择任务

            ind = torch.unsqueeze(task_op,1).tolist()

            task_seq_list.append(task_op)

            p_op,p_action_pro = self.actor2(index,task_op,p_action_pro,place_time,process_time,train)
            p_op_list.append(p_op)

            task_feas,task_mask,place_time,reward = self.env.step(task_op, p_op)
            rewards += reward


        p_action_pro = p_action_pro.view(self.batch, configs.n_j)  ##(batch,n)

        task_action_pro = action_pro.view(self.batch, configs.n_j) ##(batch,n)

        task_seq =torch.unsqueeze(task_seq_list[0],1)

        place_seq = torch.unsqueeze(p_op_list[0],1)

        q = q.to(DEVICE)

        for i in range(configs.n_j-1):
            task_seq = torch.cat([task_seq, torch.unsqueeze(task_seq_list[i + 1],1)], dim=1)
        for i in range(configs.n_j-1):
            place_seq = torch.cat([place_seq, torch.unsqueeze(p_op_list[i + 1],1)], dim=1)

        # print(task_seq[0])

        rewards = torch.from_numpy(rewards).to(DEVICE)
        rewards = rewards.to(torch.float32)

        return task_seq,place_seq,task_action_pro,p_action_pro,rewards

    def updata(self,task_action_pro,reward1,q,lr):

        opt = optim.Adam(self.actor1.parameters(), lr)
        pro = torch.log(task_action_pro)

        loss = torch.sum(pro, dim=1)

        score = reward1 - q
        score = score.detach()

        loss = score * loss
        loss = torch.sum(loss) / configs.batch

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor1.parameters(), max_norm=0.5)
        opt.step()

    def updata2(self, p_action_pro, reward1, q,lr):

        opt = optim.Adam(self.actor2.parameters(), lr)

        pro = torch.log(p_action_pro)

        loss = torch.sum(pro, dim=1).to(DEVICE)
        score = reward1 - q
        score = score.detach().to(DEVICE)

        loss = score * loss
        loss = torch.sum(loss) / configs.batch

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor2.parameters(), max_norm=0.5)
        opt.step()



