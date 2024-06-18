import numpy as np
from Params import configs
from Data import getdata

from Data import data

def traindata():#  生成用于模型训练的训练数据集
    datas = data(configs.fill,
                 configs.filo,
                 configs.fis,
                 configs.ci,
                 configs.B,
                 configs.w,
                 configs.p,
                 configs.ps,
                 configs.g,
                 configs.time,
                 configs.batch,
                 configs.n_j,
                 )
    # print(datas.shape)
    datas = np.array(datas)
    # print(datas[0].shape)
    ds = datas[0].reshape((configs.time, configs.batch, configs.n_j))
    T = datas[1].reshape((configs.time, configs.batch, configs.n_j))
    tils = datas[2].reshape((configs.time, configs.batch, configs.n_j))
    # print(tils[0])
    ties = datas[3].reshape((configs.time, configs.batch, configs.n_j))
    # print(ties.shape)
    tiss = datas[4].reshape((configs.time, configs.batch, configs.n_j))
    tises = datas[5].reshape((configs.time, configs.batch, configs.n_j))
    tisas = datas[6].reshape((configs.time, configs.batch, configs.n_j))
    datas = np.concatenate((ds, T, tils, ties, tiss,tises,tisas), axis=1)
    datas = datas.reshape(configs.time, -1, configs.batch, configs.n_j)
    datas = list(datas)
    # print(datas)

    np.save('data2//20//compare1//datas{}_1000_2000.npy'.format(configs.n_j), datas)

def data2(): # 生成用于模型训练的测试数据集
    testdatas = data(configs.fill,
                     configs.filo,
                     configs.fis,
                     configs.ci,
                     configs.B,
                     configs.w,
                     configs.p,
                     configs.ps,
                     configs.g,
                     configs.testtime,
                     configs.batch,
                     configs.n_j,
                    )
    testdatas = np.array(testdatas)
    ds = testdatas[0].reshape((configs.testtime, configs.batch, configs.n_j))
    T = testdatas[1].reshape((configs.testtime, configs.batch, configs.n_j))
    tils = testdatas[2].reshape((configs.testtime, configs.batch, configs.n_j))
    ties = testdatas[3].reshape((configs.testtime, configs.batch, configs.n_j))
    tiss = testdatas[4].reshape((configs.testtime, configs.batch, configs.n_j))
    tises = testdatas[5].reshape((configs.testtime, configs.batch, configs.n_j))
    tisas = testdatas[6].reshape((configs.testtime, configs.batch, configs.n_j))
    testdatas = np.concatenate((ds, T, tils, ties, tiss, tises, tisas), axis=1)
    testdatas = testdatas.reshape(configs.testtime, -1, configs.batch, configs.n_j)
    testdatas = list(testdatas)
    # for i in range(configs.testtime):
    #     datas.append(data(configs.batch, configs.n_j))
    # np.save('testdatas13_1000_2000.npy', testdatas)
    np.save('data2//20//compare1//testdatas{}_1000_2000.npy'.format(configs.n_j), testdatas)


def data3(): #  生成用于模型训练的验证数据集
    comtestdatas = data(configs.fill,
                     configs.filo,
                     configs.fis,
                     configs.ci,
                     configs.B,
                     configs.w,
                     configs.p,
                     configs.ps,
                     configs.g,
                     configs.comtesttime,
                     configs.batch,
                     configs.n_j,
                    )
    comtestdatas = np.array(comtestdatas)
    ds = comtestdatas[0].reshape((configs.comtesttime, configs.batch, configs.n_j))
    T = comtestdatas[1].reshape((configs.comtesttime, configs.batch, configs.n_j))
    tils = comtestdatas[2].reshape((configs.comtesttime, configs.batch, configs.n_j))
    ties = comtestdatas[3].reshape((configs.comtesttime, configs.batch, configs.n_j))
    tiss = comtestdatas[4].reshape((configs.comtesttime, configs.batch, configs.n_j))
    tises = comtestdatas[5].reshape((configs.comtesttime, configs.batch, configs.n_j))
    tisas = comtestdatas[6].reshape((configs.comtesttime, configs.batch, configs.n_j))
    comtestdatas = np.concatenate((ds, T, tils, ties, tiss, tises, tisas), axis=1)
    comtestdatas = comtestdatas.reshape(configs.comtesttime, -1, configs.batch, configs.n_j)
    comtestdatas = list(comtestdatas)
    #     datas.append(data(configs.batch, configs.n_j))
    # np.save('com_testdatas13_1000_2000.npy', testdatas)
    np.save('data2//20//compare1//com_testdatas{}_1000_2000.npy'.format(configs.n_j), comtestdatas)


np.random.seed(11)
traindata()
# data2()
# data3()