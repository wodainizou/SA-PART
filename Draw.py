#  根据训练过程中保存的数据绘制模型的收敛曲线

import matplotlib.pyplot
import matplotlib.pyplot as plt
import re
from Params import configs
import numpy as np


def Draw():
    size = '1000_2000'
    compare = 1
    max = configs.maxtask

    f = open('.//train_vali//{}//compare{}//{}_{}.txt'.format(configs.n_j,compare,configs.n_j,max), 'r')


    data = []

    for line in f:
        data.append(re.sub('\D', '', line))
    print(data)

    datas = []
    for i in data:
        datas.append(int(i) / 100000)
    f.close()

    print(datas)

    fig, ax = plt.subplots()
    plt.rc('font', family='Times New Roman', size=12)

    x = np.arange(1, len(datas) + 1)
    y = datas

    plt.plot(x, y, c='dimgray', marker='.', ms=0, label="a")

    plt.xlabel("Iterations", fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel("Time Delay", fontdict={'family': 'Times New Roman', 'size': 18})
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)

    ax.set_facecolor("whitesmoke")
    plt.legend(['Loss Function'], prop={'family': 'Times New Roman', 'size': 14})

    plt.grid(True, alpha=0.5)

    # ax.spines['top'].set_visible(False)  # remove up
    # ax.spines['right'].set_visible(False)  # remove down

    plt.savefig("10_4p.jpg", dpi=600)
    plt.show()

Draw()