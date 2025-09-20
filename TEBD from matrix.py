import numpy as np

from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigs
from numpy import linalg as LA
from scipy.linalg import expm
# from typing import Optional

shape=(2,2,2,2,2,2) #创建六阶张量
tensor_6d = np.zeros(shape)#构建全为0的初始张量
# print("张量形状:", tensor_6d.shape)#打印张量的形状
tensor_6d[0,0,0,0,0,0]=1#设定初始处于所有自旋都朝上的状态
# tensor_6d[0,0,0,1,0,1]=1
dims = tensor_6d.shape  # 获取张量形状


sX = np.array([[0, 1], [1, 0]])
sZ = np.array([[1, 0], [0, -1]])
I=np.array([[1, 0], [0, 1]])#构造泡利矩阵
hamZ = ((-np.kron(sZ, sZ))).reshape(2, 2, 2, 2)#构建local Hamiltonian
hamX = (sX).reshape(2, 2)#每个格点上的\sigma x作用
# print(sX)
# print(hamZ)
d=2
tau=0.01#更新时间间隔

gateZ = expm(-1* tau * hamZ.reshape(d**2, d**2)).reshape(d, d, d, d)
gateX = expm(-1* tau * hamX.reshape(d,d)).reshape(d, d)#对于相互作用取e指数再重构成为四阶张量
# print(gateZ)

times=100#定义时间间隔作用次数，也即有times*tau=T(总时间)
psi=[]#创建一个空的函数列表
current_state=tensor_6d.copy()
# print(current_state)#初始化为设定的张量

for j in range(times):

    for i in range(tensor_6d.ndim-1):#对\sigma z门进行应用

        new_shape=(2**i,2,2,2**(tensor_6d.ndim-2-i))

        reshaped_state=current_state.reshape(new_shape)#对初始化态进行重构

        evolved_state = ncon([reshaped_state, gateZ],
                             [[-1, 1, 2, -4], [-2, -3, 1, 2]])#把重构的态与\sigma z 门进行缩并

        current_state = evolved_state.reshape(shape)# 恢复原始形状

    for k in range(tensor_6d.ndim):# 对每个量子比特应用横场,二者分开进行作用主要原因是
        # 因为\sigma z与\sigma x不对易

        new_shape=(2**k,2,2**(tensor_6d.ndim-1-k))#每次只作用到一个自旋上
        reshaped_state = current_state.reshape(new_shape)

        evolved_state = ncon([reshaped_state, gateX],
                             [[-1,1,-3],[-2,1]])
        current_state = evolved_state.reshape(shape)  # 恢复原始形状

    current_state=current_state.reshape(2**tensor_6d.ndim)/LA.norm(current_state)#对矢量进行归一化

    psi.append(current_state)


print(psi[-1])
print(psi[-2])



