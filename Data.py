# 根据论文中的参数生成相关数据

import numpy as np
from scipy.stats import rice

def Datageneration(time,batch,n):
    data_size = np.random.randint(1000, 2000, (int(time*batch),1, n))
    data_size = data_size.squeeze()
    data_size.squeeze()

    D = np.random.randint(1345817.4925,1363628.24846,(int(time*batch),1, n))
    D = D.squeeze()

    T_ = np.random.uniform(0.5, 1, (int(time*batch),1, n))
    T_ = T_.squeeze()
    return data_size,T_,D

def vvs(D,B,w,p,g):
    ricean_var = rice(b=8.4, loc=0, scale=7.18)
    ρ_is = ricean_var.rvs(size=1)

    z1 = (4 * 0.33 * 30 * 10**9) / (3 * 10**8)
    Gr = np.power(z1,2)
    x1 = (4 * 3.14 * D * 30 * 10**9)/ (3 * 10**8)
    h_is = ρ_is / x1
    H_is = np.power(h_is,2)
    a = p * g * Gr * H_is
    v_is = B * 10**6 * np.log2(1+a/w)
    return v_is

def vss(D,B,w,ps):
    ricean_var = rice(b=8.4, loc=0, scale=7.18)
    ρ_Gs = ricean_var.rvs(size=1)

    z2 = (4 * 0.5 * 20 * 10**9) / (3 * 10**8)
    Gt = np.power(z2,2)
    z3 = (4 * 7.3 * 20 * 10**9) / (3 * 10**8)
    Gg = np.power(z3, 2)
    x2 = (4 * 3.14 * D * 20 * 10**9) / (3 * 10**8)
    h_Gs = ρ_Gs / x2
    H_Gs = np.power(h_Gs, 2)
    d = ps * Gt * Gg * H_Gs
    v_Gs = B * 10**6 * np.log2(1 + d/w)
    return v_Gs

def getdata(n_j,fill,filo,fis,ci,B,w,p,ps,g,time,batch):
    fis = fis * 10**9
    fill = np.random.randint(1, fill + 1, (n_j))
    fie = fill.astype('float32') * 10**9
    filo = np.random.randint(1, filo + 1, (n_j))
    fil = filo.astype('float32') * 10**9

    til = []
    tie = []
    tisa = []
    v_is = []
    v_Gs = []

    taskdata = Datageneration(time,batch,n_j)
    datasize = taskdata[0]
    D = taskdata[2]

    v_is = vvs(D,B,w,p,g)
    v_Gs = vss(D,B,w,ps)

    T = taskdata[1]

    for i in datasize:
        til.append(i*ci*1000/fil)
    for i in datasize:
        tie.append(i*ci*1000/fie)
    for i in datasize:
        tisa.append(i*ci*1000/fis)

    tis_up = datasize * 1000/v_is
    tis_down = datasize * 1000/v_Gs
    t_pro = D/(3 * 10**8)
    tis = tis_up + t_pro
    tise = tis_up + tis_down + t_pro + t_pro

    # print(tis)
    return datasize,T,til,tie,tise,tisa,tis
# getdata(n_j=10,batch=3,time=2,fil=1,fie=10,fis=15,ci=1000,B=800,w=0.000000001)

def data(fill,filo,fis,ci,B,w,p,ps,g,time,batch,n_j):
    datasizes = []
    Ts = []
    tils = []

    ties = []
    tisas = []
    tiss = []
    tises = []
    getdatas = getdata(n_j,fill,filo,fis,ci,B,w,p,ps,g,time,batch)
    datasizes.append(getdatas[0])
    Ts.append(getdatas[1])
    tils.append(getdatas[2])
    ties.append(getdatas[3])
    tiss.append(getdatas[4])
    tises.append(getdatas[5])
    tisas.append(getdatas[-1])
    return datasizes, Ts, tils, ties, tiss, tises, tisas



