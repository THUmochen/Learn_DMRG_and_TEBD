# doApplyHam.py
# ---------------------------------------------------------------------
# Routine used in the implementation of exact diagonalization.
#
# by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 6/2019

import numpy as np
from scipy.linalg import expm

tau = 0.01

def doApplyHam(psiIn, hloc, N, usePBC):
    d = 2

    psiOut = np.zeros(psiIn.size)
    # 应用键相互作用和外场项（通过键添加的部分）
    for k in range(N - 1):

        psiOut += np.tensordot(hloc.reshape(d**2, d**2),
                               psiIn.reshape(d**k, d**2, d**(N - 2 - k)),
                               axes=[[1], [1]]).transpose(1, 0, 2).reshape(d**N)
    # 额外添加端点格点的外场项：由于键添加的外场项只给了端点0.5的系数，这里补上剩余的0.5
    # 但注意：当前hloc的外场项系数是0.5，所以总外场项需要额外添加每个端点0.5的σ_x
    if not usePBC:

        # 添加右端点格点的外场项：格点N-1
        h_right = sx
        gatelast=expm(- tau *h_right.reshape(d,d) ).reshape(d, d)
        psiOut += np.tensordot(gatelast, psiIn.reshape(d**(N-1), d), axes=[[1], [1]]).transpose(1,0).reshape(d**N)
    return psiOut


"""
mainExactDiag.py
---------------------------------------------------------------------
Script file for initializing exact diagonalization using the 'eigsh' routine
for a 1D quantum system.

"""

from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer#可以用于测量代码执行时间


#设置参数
Nsites = 6  # 自旋数目选取
usePBC = False  # 是否考虑周期性边界条件
numval = 1  # 求解的本征态数目

d = 2  # 每一个自旋的维度
sx = np.array([[0, 1.0], [1.0, 0]])
sz = np.array([[1.0, 0], [0, -1.0]])
sI = np.array([[1.0, 0], [0, 1.0]])#分别定义泡利矩阵

hloc = (-np.kron(sz, sz)+np.kron(sx, sI)
          ).reshape(2, 2, 2, 2)#构造局域哈密顿量，其中kron为直积
holc=expm(-tau*hloc.reshape(d**2,d**2)).reshape(2,2,2, 2)
EnExact = -2 / np.sin(np.pi / (2 * Nsites))  # Note: only for PBC

print(EnExact)
def doApplyHamClosed(psiIn):#将输入波函数应用到doApplyHam函数当中
  return doApplyHam(psiIn, hloc, Nsites, usePBC)


H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)

start_time = timer()
Energy, psi = eigsh(H, k=numval, which='SA')#精确对角化函数
diag_time = timer() - start_time

EnErr = Energy[0] - EnExact  # should equal to zero
print('NumSites: %d, Time: %1.2f, Energy: %e, EnErr: %e' %
      (Nsites, diag_time, Energy[0], EnErr))
print(psi)
