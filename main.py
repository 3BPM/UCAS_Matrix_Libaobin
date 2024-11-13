import os,argparse
import numpy as np
import random
from numpy.linalg import matrix_rank
from math import sqrt
def vector_norm(vector):
    return sqrt(np.dot(vector, vector))
def normalize_vector(vector):
    # 计算向量的模长
    norm = vector_norm(vector)
    if norm == 0:# 如果模长为0，返回原0向量
        return vector
    return vector / norm
#modified施密特正交化实现的QR分解
def QR_Factor(A,isprint=False):
    m, n = A.shape
    Q = np.copy(A)
    E=[]
    R = np.zeros([n, n], dtype='float64')
    norm=vector_norm(Q[:, 0])
    Q[:, 0] = normalize_vector(Q[:, 0])
    R[0, 0] = norm
    R[0,1:]= (Q[:, 0] .T)@A[:, 1:]
    print(f"第1步，Q:{Q}\n R:{R}\n")
    for k in range(1,n):
        uk_last=Q[:, [k-1]]
        uk=A[:, k]
        Ek=np.eye(n)-uk_last@uk_last.T
        E.append(Ek)
        for i in range(k):#应该还有更简便的完全按照模和内积43页
            uk=E[i]@uk
        norm=vector_norm(uk)
        uk=normalize_vector(uk)
        Q[:, k]=uk
        R[k, k] = norm
        R[k,k+1:]= (uk.T)@A[:, k+1:]

        print(f"第{k+1}步，Q:{Q}\n R:{R}\n")
    return Q, R
def create_Tpq_A(n, p, q, c, s):
    T_pq = np.eye(n)#, dtype=complex
    # 更新矩阵的指定位置
    T_pq[p, p] = np.conj(c)  # 对角线位置 (p, p) 为 c 的共轭
    T_pq[q, q] = c           # 对角线位置 (q, q) 为 c
    T_pq[p, q] = np.conj(s)  # 位置 (p, q) 为 s 的共轭
    T_pq[q, p] = -s          # 位置 (q, p) 为 -s
    return T_pq
def Givens_Reduction(A):
    #矩阵运算都用numpy来处理
    A=np.array(A)
    (m,n)=A.shape
    #首先初始化P，T矩阵
    Q=np.identity(m)
    R=np.copy(A)
    #按顺序每一列进行变换
    for columnj in range(n):
        for rowi in range(m-1,columnj,-1):
            #首先得到要变换的两个数
            x=R[columnj,columnj]
            y=R[rowi,columnj]
            norm=np.sqrt(x**2+y**2)
            #直接构造旋转矩阵
            c=x/norm
            s=y/norm
            P_ij=create_Tpq_A(m,columnj,rowi,c,s)
            #累乘可以得到最终的P和T矩阵
            Q=P_ij@Q
            R=P_ij@R
            print(f"第i步骤P({rowi},{columnj})={P_ij}\n R:={R}\n ")

    Q_A=Q.T
    return Q_A,R



def Householder_Reduction(A,isprint=True):
    A=np.array(A).astype(np.float32)
    (len_r,len_c)=A.shape

    Q=np.identity(len_r)
    #分别处理两种不:方阵的情况
    if len_r > len_c:
        iter_num=len_c
    else:
        iter_num=len_r-1

    for idx in range(iter_num):
        #首先构造ei向量
        e = np.zeros(len_r-idx)
        e[0] = 1
        I = np.identity(len_r-idx)
        #注意u要:列向量
        a = A[idx:,idx]#每行向量
        u = A[idx:,idx].T - vector_norm(a)*e.T
        u = normalize_vector(u)
        exp_u = np.expand_dims(u,axis=0)
        #得到反射算子
        Reflection = I - 2.0*exp_u.T@exp_u
        #拓展成正常大小
        Hi = np.identity(len_r)
        Hi[idx:,idx:] = Reflection
        #更新
        Q = Hi@Q
        A = Hi@A
        if isprint:
            print(f"第{idx}步，u:{u},\nReflection:{Reflection}\nQ{Q}\nHA:{A}\n")

    Q_A=Q.T
    R_A=A
    return Q_A,R_A





#判断方程Ax=b:否有解
def is_solvable(A,b):
    Ab=np.concatenate((A,np.expand_dims(b,axis=0).T),axis=1)
    #如果增广矩阵的秩等于A的秩就有解
    return matrix_rank(A)==matrix_rank(Ab)

def solve_A(Q,R,b):
    n=R.shape[1]
    #等价于求解 Rx=Q.Tb
    Q_T_b = Q.T@ b
    #去掉R多余的行
    if R.shape[0]!=R.shape[1]:
        R=np.delete(R,[R.shape[1],R.shape[0]-1],axis=0)
        Q_T_b=Q_T_b[:R.shape[1]]
    #回代法求解Rx=Q.Tb,上三角
    if is_solvable(R,Q_T_b):#1.方程有解
        x=np.copy(Q_T_b)
        has_infinite_solutions = False
        for i in range(n-1,-1,-1):#从n行到第1行
            x[i]=Q_T_b[i]#得到等式和
            for j in range(n-1,i,-1):
                x[i] =x[i]- R[i][j]*x[j]#减去前面的分量
            if int(R[i][i])==0:#2.如果R对角线存在0，那就:有无穷解
                # 随便给出一个数值解
                x[i] = -999.0
                has_infinite_solutions = True
            else:#3.唯一解
                x[i] = float(x[i])/R[i][i]#注意U的对角不:1

        if has_infinite_solutions:
            print('无穷解')
        else:
            print('唯一解')
    else:#方程无解
        print('方程无解')
    return x



ap = argparse.ArgumentParser(description="""完成课堂上讲的关于矩阵正交分解的程序实现，包括Modified Gram-Schmid方法，Houshold reduction和Givens reduction方法，要求如下：

        1、一个综合程序，根据选择参数的不同，实现不同方法的矩阵分解；在此基础上，实现Ax=b方程组的求解；

         2、可以用matlab、Python等编写程序，需附上简单的程序说明，比如参数代表什么意思，输入什么，输出什么等等，附上相应的例子；

         3、一定:可执行文件，例如 .m文件等,不能:word或者txt文档。附上源代码，不能为直接调用matlab等函数库;""")
#选择使用的分解算法，可选项'LU','QR','HR','GR','URV'
ap.add_argument("--model", type=str, choices=['Schmid','Houshold','Givens'], default="Schmid",
                help="包括Modified Gram-Schmid方法，Houshold reduction和Givens reduction方法, \
                     Schmid','Houshold','Givens',")
ap.add_argument("--precision", type=str, default="20",
                help="小数点精度")
args = ap.parse_args()


if __name__ == "__main__":
    #控制小数点精度
    np.set_printoptions(precision=int(args.precision),suppress=True)

    A1=np.array([
    [1,19,-34],
    [-2,-5,20],
    [2,8,37]
    ], dtype='float64')
    A=np.array([
    [1,1,1],
    [0.001,0.001,0],
    [0.001,0,0.001]
    ], dtype='float64')
    A=np.array([
    [0,-20,-14],
    [3,27,-4],
    [4,11,-2]
    ], dtype='float64')
    b=np.array([100,0.01,0.05], dtype='float64')
    m, n = A.shape
    if A.size == 0:
        raise ValueError("不能:空矩阵")
    print("A矩阵:\n",A,"\n",args.model,"方法")
    rank = np.linalg.matrix_rank(A)
    if rank < min(m,n):
        raise ValueError("警告: 矩阵不:满秩的! 矩阵的秩为:", rank)
    else:
        print("矩阵:满秩的, 秩为:", rank)
    if args.model == "Schmid":
        Q, R = QR_Factor(A)
    elif args.model == "Houshold":
        Q, R = Householder_Reduction(A)
    elif args.model == "Givens":
        Q, R = Givens_Reduction(A)
    print("检查QxR",Q@R)
    x = solve_A(Q,R, b)
    print(" x:")
    print(x)
    x = np.linalg.solve(A, b)
    print("检查x",x)

