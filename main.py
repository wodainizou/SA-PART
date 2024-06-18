#  模型训练的主要功能

from Params import configs
import torch
import os
import numpy as np
from Actor_ctitic import actor_critic


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

compare = 1
size = '1000_2000'

 #  下载训练数据和测试数据
datas = np.load('data//{}//compare{}//datas{}_{}.npy'.format(configs.n_j,compare,configs.n_j,size))
datas.astype('float16')
# print(datas.dtype)
testdatas = np.load('data//{}//compare{}//com_testdatas{}_{}.npy'.format(configs.n_j,compare,configs.n_j,size))

Net1 = actor_critic(batch=configs.batch,
                    hidden_dim = configs.hidden_dim,
                    M=8,
                    device=configs.device).to(DEVICE)

Net2 = actor_critic(batch=configs.batch,
                    hidden_dim = configs.hidden_dim,
                    M=8,
                    device=configs.device).to(DEVICE)

Net2.actor2.load_state_dict(Net1.actor2.state_dict())


min = 50000000000

if configs.batch == 24:
    lr = 0.000001
    print('lr=',lr)
elif configs.batch == 8:
    lr = 0.0000001
    print('lr=', lr)

bl_alpha = 0.05

output_dir = 'train_process//{}//compare{}'.format(configs.n_j,compare)

save_dir = os.path.join(os.getcwd(), output_dir)


from scipy.stats import ttest_rel

contintrain = 0
Net2.load_state_dict(Net1.state_dict())


for epoch in range(configs.epochs):
    for i in range(configs.time):
        data = datas[i]
        # print(data.shape)
        task_seq,p_seq, task_action_pro, p_action_pro, reward1 = Net1(data, 1)
        _,_,_,_,reward2 = Net2(data, 1)

        reward1 = reward1.detach()
        torch.cuda.empty_cache()

        Net1.updata(task_action_pro, reward1, reward2,lr)
        Net1.updata2(p_action_pro, reward1, reward2,lr)
        print('epoch={},i={},time1={},time2={}'.format(epoch, i, torch.mean(reward1),
                                                       torch.mean(reward2)))
        reward1 = reward1

        with torch.no_grad():

            temp = reward1.mean() - reward2.mean()
            if temp < 0:
                tt, pp = ttest_rel(reward1.cpu().numpy(), reward2.cpu().numpy())
                p_val = pp / 2

                assert tt < 0, "T-statistic should be negative"

                if p_val < bl_alpha:
                    print('Update baseline')
                    Net2.load_state_dict(Net1.state_dict())

            """Every 20 iterations check whether the model needs to save parameters"""

            if i % 20 == 0:

                length = torch.zeros(1).to(DEVICE)

                for j in range(configs.comtesttime):
                    torch.cuda.empty_cache()
                    _,_, _, _, r = Net1(testdatas[j], 0)

                    length = length + torch.mean(r)

                length = length / configs.comtesttime

                if length < min:
                    torch.save(Net1.state_dict(), os.path.join(save_dir,
                                                               'epoch{}-i{}-dis_{:.5f}.pt'.format(
                                                                   epoch, i, length.item())))

                    torch.save(Net1.state_dict(), os.path.join(save_dir,
                                                               'actor{}_mutil_actor.pt'.format(configs.n_j)))

                    min = length

                file_writing_obj1 = open('./train_vali/{}//compare{}//{}_{}.txt'.format(configs.n_j, compare,configs.n_j,configs.maxtask),
                                         'a')
                file_writing_obj1.writelines(str(length) + '\n')

                print('length=', length.item(), 'min=', min.item())

                file_writing_obj1.close()