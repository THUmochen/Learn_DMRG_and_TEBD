import numpy as np
from numpy import linalg as LA
from ncon import ncon
shape=(2,2,2,2,2,2) #创建六阶张量
tensor_6d = np.zeros(shape)#构建全为0的初始张量
# print("张量形状:", tensor_6d.shape)#打印张量的形状
tensor_6d[0,0,0,0,0,1]=1#设定初始处于所有自旋都朝上的状态
tensor_6d[0,0,0,1,0,1]=1
dims = tensor_6d.shape  # 获取张量形状
#进行构建MPS
def mps_decomposition(tensor_6d, bond_dims=None):#函数语法学习？

    if bond_dims is None:
        bond_dims = [min(np.prod(dims[:i]), np.prod(dims[i:])) for i in range(1, 6) ]#.prod语法学习？
#设置键指标，根据SVD算法，选取两边的指标最小为键指标

    cores = []#初始化核心张量列表
    matrix = tensor_6d.reshape(dims[0], -1)#把张量第一个指标作为矩阵行指标，其余指标乘积作为列指标。
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)#进行SVD分解
    bond_dim = min(bond_dims[0], s.shape[0])#判断是否这里的奇异值矩阵是否类似满秩的形式
    u=u[:,:bond_dim ]#保留对应s奇异值对应的那些类似的本征列矢
    s=s[:bond_dim]
    vh=vh[:bond_dim,:]#保留对应s奇异值对应的那些类似的本征横矢
    core1=u.reshape(dims[0], bond_dim)
    cores.append(core1)#把u添加到核心张量列表当中
    matrix = np.diag(s) @ vh#把s形成对角阵与vh直接矩阵相乘
    # print(cores[0])

    for i in range(1,5):
       matrix = matrix.reshape(bond_dim*dims[i],-1)#注意这里此时左端等距张量有两条自由腿，一个键张量
       ui, si, vhi = np.linalg.svd(matrix, full_matrices=False)
       bond_dim_=bond_dim
       bond_dim = min(bond_dims[i], si.shape[0])
       ui=ui[:,:bond_dim ]
       si=si[:bond_dim]
       vhi=vhi[:bond_dim,:]
       core2=ui.reshape(bond_dim_,dims[i],bond_dim)
       cores.append(core2)#完全相同的构造方法
       matrix = np.diag(si) @ vhi

    core6 = matrix.reshape(bond_dim, dims[5])#最后一个是一个矩阵，和上面的三阶张量不同，无法放入循环，手动加入。
    cores.append(core6)
    return cores
# print(mps_decomposition(tensor_6d, bond_dims=None))

mps=mps_decomposition(tensor_6d, bond_dims=None)#把结果给到mps当中

print(mps[3])
I=np.eye(2)#定义单位矩阵
sz=np.array([[1,0],[0,-1]])#定义泡利z矩阵
sx = np.array([[0, 1], [1, 0]])  # 泡利X矩阵 (用于外场项)
#进行一维伊辛链哈密顿量MPO的构造。
n_sites=6
def create_ising_mpo(n_sites, J=-1.0, h=1):#这里表示自旋数目为n个，伊辛相互作用耦合常数为1，无外场
    I=np.eye(2)#定义单位矩阵
    sz=np.array([[1,0],[0,-1]])#定义泡利z矩阵
    sx = np.array([[0, 1], [1, 0]])  # 泡利X矩阵 (用于外场项)
    mpo = []#初始化MPO列表

    bond_dim=3 #
    W_first = np.zeros((1, bond_dim, 2, 2))#四维张量左边界第一个，四个指标分别是左键，右键，物理输入，物理输出
    W_first[0, 0, :, :] = I  # 单位算符
    W_first[0, 1, :, :] = J*sz  # 耦合项
    if h != 0:#调整h为0，则可不考虑外场求解
        W_first[0, 2, :, :] = h * sx  # 外场项
    mpo.append(W_first)
    for i in range(1, n_sites - 1):
        W_mid = np.zeros((bond_dim, bond_dim, 2, 2))
        W_mid[0, 0, :, :] = I  # 单位算符
        W_mid[1, 0, :, :] = sz  # 与左边泡利算符的耦合项
        W_mid[0, 1, :, :] =  sz  # 与右边泡利算符的耦合项
        if h != 0:
            W_mid[0, 2, :, :] = h * sx  # 外场项
            W_mid[2, 2, :, :] = I  # 保持外场项
        mpo.append(W_mid)
    W_last = np.zeros((bond_dim, 1, 2, 2))#最后一项只有一个键指标
    W_last[0, 0, :, :] = I
    W_last[1, 0, :, :] = sz
    if h != 0:
        W_last[2, 0, :, :] = sx  # 外场项
    mpo.append(W_last)
    return mpo
# print(create_ising_mpo(n_sites, J=1.0, h=0.0))
mpo=create_ising_mpo(n_sites, J=-1.0, h=1)#把结果幅到mpo当中

#对mpo各个进行联合形成一个哈密顿量矩阵
k=len(mpo)
print(k)
H=mpo[0].reshape(mpo[0].shape[1],2,2)
H=np.transpose(H,(1,2,0))
print(H.shape)
for m in range(1,k):
    H_term=np.transpose(mpo[m],(0,2,3,1))
    H=ncon([H,mpo[m]],[[-1,-2,1],[1,-3,-4,-5]])
    H=np.transpose(H,(0,2,1,3,4)).reshape(2**(m+1),2**(m+1),mpo[m].shape[1])

H=H.reshape(2**k,2**k)
print(H)

eigenvalues = np.linalg.eigvalsh(H)
print("基态能量:", eigenvalues[0])
# import numpy as np
# d = 2  # 每一个自旋的维度
# sx = np.array([[0, 1.0], [1.0, 0]])
# sz = np.array([[1.0, 0], [0, -1.0]])
# sI = np.array([[1.0, 0], [0, 1.0]])#分别定义泡利矩阵
#
# hloc = (-np.kron(sz, sz)+ 0.5 * np.kron(sx, sI) + 0.5 * np.kron(sI, sx)
#           ).reshape(2, 2, 2, 2)#构造局域哈密顿量，其中kron为直积
# print(hloc)
#
# #构造MPO
# H=(-np.kron(sz, sz,I,I,I,I)-np.kron(I, sz,sz,I,I,I)-np.kron(I, I,sz,sz,I,I)-np.kron(I,I,I,sz, sz,I)
#    -np.kron(I,I,I,I,sz, sz)+np.kron(sx,I,I,I,I,I)+np.kron(I,sx,I,I,I,I)+np.kron(I,I,sx,I,I,I)
#    +np.kron(I,I,I,sx,I,I)+np.kron(I,I,I,I,sx,I)+np.kron(I,I,I,I,I,sx))
# # import numpy as np
# # a=[1,2,3]
# # print(np.diag(a))
#
# n_sites=6
# sI = np.eye(2)  # 单位矩阵
# sz = np.array([[1, 0], [0, -1]])#Pauli Z矩阵
# sx = np.array([[0, 1], [1, 0]]) #Pauli X矩阵
# # 连续计算直积态的函数
# def multi_kron(*matrices):
#     result = matrices[0]#把result初始化为第一个矩阵
#     for mat in matrices[1:]:#从第二个开始连续进行直积操作
#         result = np.kron(result, mat)
#     return result
# # 哈密顿量H集合
# H_terms = [
#     -multi_kron(sz, sz, sI, sI, sI, sI),
#     -multi_kron(sI, sz, sz, sI, sI, sI),
#     -multi_kron(sI, sI, sz, sz, sI, sI),
#     -multi_kron(sI, sI, sI, sz, sz, sI),
#     -multi_kron(sI, sI, sI, sI, sz, sz),
#     multi_kron(sx, sI, sI, sI, sI, sI),
#     multi_kron(sI, sx, sI, sI, sI, sI),
#     multi_kron(sI, sI, sx, sI, sI, sI),
#     multi_kron(sI, sI, sI, sx, sI, sI),
#     multi_kron(sI, sI, sI, sI, sx, sI),
#     multi_kron(sI, sI, sI, sI, sI, sx)]
# # 将所有项相加得到总哈密顿量
# H = sum(H_terms)
# mpo=[]#初始化mpo序列
#
# bond_dims=[]
# for i in range(n_sites-1):
#     bond_dims .append (min(2 ** (2 * (i + 1)), 2 ** (2 * (n_sites - 1 - i))))#构建键指标
# # print(bond_dims[0], bond_dims[2])检验键指标是否构建正确
#
# H=H.reshape(2**2,2**(2*n_sites-2))#给哈密顿量做成
# u, s, vh = np.linalg.svd(H, full_matrices=False)  # 进行SVD分解
#
# bond_dim=min(bond_dims[0], s.shape[0])#更新键指标
# u = u[:, :bond_dim]
# s = s[:bond_dim]
# vh=vh[:bond_dim, :]
# u_=u.T#由于mpo的键指标一般占据前两个指标，因此转置后便于进行重构成为mpo
# # print(u_)
# # print(u)
#
# mpo.append(u_.reshape(1,bond_dim,2,2))
# print(mpo[0].shape )
# H=np.diag(s) @ vh#重新形成剩余的哈密顿量
# print(H.shape)
# #
# # for i in range(n_sites-2):
# H_=H.reshape(bond_dim*2,2**(n_sites-i-1),2,2**(n_sites-i-1))
# print(H_.shape)
#     # H=np.transpose(H_,(0,2,1,3)).reshape(bond_dim*4,2**(2*n_sites-2*i-2))
#

import numpy as np

# 创建一个3x4x5的张量（三维数组）
tensor = np.zeros((3, 4, 5))

# 创建一个向量
vector = np.array([1, 2, 3, 4, 5])

# 将向量赋值到张量的特定位置
# 例如，赋值到第一个维度的索引1，第二个维度的索引2
tensor[1, 2, :] = vector  # 将向量赋值到张量的这个切片

print(tensor[1, 2, :])  # 输出: [1. 2. 3. 4. 5.]







