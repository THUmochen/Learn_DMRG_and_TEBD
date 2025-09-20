import numpy as np
from scipy.linalg import expm
from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer

# 设置参数
Nsites = 16  # 自旋数目选取
usePBC = False  # 是否考虑周期性边界条件
numval = 1  # 求解的本征态数目

d = 2  # 每一个自旋的维度
sx = np.array([[0, 1.0], [1.0, 0]])
sz = np.array([[1.0, 0], [0, -1.0]])
sI = np.array([[1.0, 0], [0, 1.0]])  # 分别定义泡利矩阵
sy = np.array([[0, -1j], [1j, 0]])

# 修正哈密顿量定义（这里使用海森堡模型）
# 注意：sy 是复数矩阵，所以 hloc 也会是复数
hloc = -(np.kron(sx, sx) + np.kron(sy, sy) ).reshape(2, 2, 2, 2)

# 精确解（暂时设为None，因为公式可能不正确）
EnExact = None  # -2 / np.sin(np.pi / (2 * Nsites))


def doApplyHam(psiIn, hloc, N, usePBC):
    d = 2
    # 初始化复数数组，而不是实数数组
    psiOut = np.zeros(psiIn.size, dtype=complex)

    # 应用键相互作用
    for k in range(N - 1):
        # 确保所有操作都保持复数类型
        term = np.tensordot(hloc.reshape(d ** 2, d ** 2),
                            psiIn.reshape(d ** k, d ** 2, d ** (N - 2 - k)),
                            axes=[[1], [1]]).transpose(1, 0, 2).reshape(d ** N)
        psiOut += term

    # 如果使用周期性边界条件，添加最后一个键
    if usePBC:
        # 将波函数重新排列，使最后一个站点与第一个站点相邻
        psi_reshaped = psiIn.reshape(d, d ** (N - 1))
        term = np.tensordot(hloc.reshape(d ** 2, d ** 2),
                            psi_reshaped.T.reshape(d ** (N - 1), d),
                            axes=[[1], [1]]).transpose(1, 0).reshape(d ** N)
        psiOut += term

    return psiOut


def doApplyHamClosed(psiIn):  # 将输入波函数应用到doApplyHam函数当中
    return doApplyHam(psiIn, hloc, Nsites, usePBC)


# 确保使用复数类型
H = LinearOperator((2 ** Nsites, 2 ** Nsites), matvec=doApplyHamClosed, dtype=complex)

start_time = timer()
# 使用 eigsh 处理复数厄米特矩阵
Energy, psi = eigsh(H, k=numval, which='SA')  # 精确对角化函数
diag_time = timer() - start_time

print('Eigenvalue:', Energy)