import networkx as nx
import numpy as np
from math import log
from sklearn import metrics
from sklearn.cluster import KMeans
import motifcluster.motifadjacency as motif
import main_tool_new
import time

maxd = 256
# 计算NMI
def mutual_info(c_A, c_B,S):
    N_mA = len(c_A)
    N_mB = len(c_B)
    # print  "CA",c_A
    # print  "CB",c_B

    I_num = 0
    for i in range(len(c_A)):
        for j in range(len(c_B)):
            n_i = len(c_A[i])
            n_j = len(c_B[j])
            n_ij = len(set(c_A[i]) & set(c_B[j]))
            if n_ij == 0:
                continue
            log_term = log((n_ij * S * 1.0) / (n_i * n_j))

            I_num += n_ij * log_term
    I_num *= -2

    I_den = 0
    for i in range(len(c_A)):
        n_i = len(c_A[i])
        I_den += n_i * log(n_i * 1.0 / S)

    for j in range(len(c_B)):
        n_j = len(c_B[j])
        I_den += n_j * log(n_j * 1.0 / S)

    I = I_num / I_den
    return I

def get_G(filename):
    G_node = set()
    G_edge1 = []
    G_edge2 = []
    with open(filename) as f:
        G = nx.Graph()
        for line in f:
            # u, v = map(int, line.split(','))
            line = line.split('\t')
            u = int(line[0])-1
            v = int(line[1])-1
            u = u  # 从1开始要-1
            v = v
            G_node.add(u)
            G_edge1.append(u)
            G_node.add(v)
            G_edge2.append(v)
    f.close()
    G_node = list(G_node)
    G_node = sorted(G_node)
    G_node = np.array(G_node)
    for n in range(len(G_node)):
        G.add_node(n, embed=(np.random.random(maxd) - 0.5) * 2)
    for e in range(len(G_edge1)):
        index1 = np.where(G_node == G_edge1[e])[0][0]
        index2 = np.where(G_node == G_edge2[e])[0][0]
        G.add_edge(index1, index2)
    # for n in range(len(G_node)):
    #     # G.add_node(G_node[n], embed=(np.random.random(maxd)-0.5)*2)
    #     G.add_node(G_node[n], embed=(np.random.random(maxd)-0.5)*2)
    #
    # for e in range(len(G_edge1)):
    #     G.add_edge(G_edge1[e], G_edge2[e])
    return G

def get_adjMatrix(filename):
    G = get_G(filename)
    print("构建图网络完毕")
    # G = get_PubmedG(filename)
    # G = get_G_gml(filename)
    matrix = nx.adjacency_matrix(G)
    print("构建邻接矩阵完毕")
    return G,matrix

def detect_motif33(matrix):
    #检测全连接的motif，特点是邻接矩阵全为1，只需要判断上三角
    motif_matrix = np.zeros_like(matrix)
    for i in range(len(matrix)-2):
        x = np.where(matrix[i] == 1)[0]
        indexs = np.where(x > i)[0]
        for index in range(len(indexs)):
            j = x[indexs[index]]
            for index_z in range(index+1,len(indexs)):
                z = x[indexs[index_z]]
                if matrix[j][z] == 1:
                    motif_matrix[i][j] += 1
                    motif_matrix[i][z] += 1
                    motif_matrix[j][z] += 1
                    motif_matrix[j][i] += 1
                    motif_matrix[z][i] += 1
                    motif_matrix[z][j] += 1
    return motif_matrix

def detect_motif32(matrix):
    # 检测3,2的motif，x,y x,z y,z仅存在一个0
    motif_matrix = np.zeros_like(matrix)
    for i in range(len(matrix) - 2):
       for x in range(i+1,len(matrix[i])):
           if matrix[i][x] == 0:
               #若x为0了，则y,z都必须为1
                ys = np.where(matrix[i] == 1)[0]
                indexs = np.where(ys > x)[0]
                for index_y in range(len(indexs)):
                    y = ys[indexs[index_y]]
                    if matrix[x][y] == 1:
                        motif_matrix[x][y] += 1
                        motif_matrix[y][x] += 1
                        motif_matrix[i][y] += 1
                        motif_matrix[y][i] += 1
                        motif_matrix[i][x] += 1
                        motif_matrix[x][i] += 1
           else:#x为1，则y,z存在一个为0
               ys = np.where(matrix[i] == 1)[0]
               indexs = np.where(ys > x)[0]
               for index_y in range(len(indexs)):#y也为1
                   y = ys[indexs[index_y]]
                   if matrix[x][y] == 0:
                       motif_matrix[i][x] += 1
                       motif_matrix[x][i] += 1
                       motif_matrix[i][y] += 1
                       motif_matrix[y][i] += 1
                       motif_matrix[y][x] += 1
                       motif_matrix[x][y] += 1
               ys = np.where(matrix[i] == 0)[0]#y为0，则z必须为1
               indexs = np.where(ys > x)[0]
               for index_y in range(len(indexs)):
                   y = ys[indexs[index_y]]
                   if matrix[x][y] == 1:
                       motif_matrix[x][y] += 1
                       motif_matrix[y][x] += 1
                       motif_matrix[i][x] += 1
                       motif_matrix[x][i] += 1
                       motif_matrix[i][y] += 1
                       motif_matrix[y][i] += 1
    return motif_matrix

def get_newMatrix(filename):#t:超参数，用于调和低阶与高阶矩阵的融合
    print("计算motif矩阵开始！")
    time_begin = time.time()
    G,matrix = get_adjMatrix(filename)
    print("邻接矩阵构造完毕")
    motif_matrix = motif.build_motif_adjacency_matrix(matrix,"M3", "func", "mean")
    # motif_matrix = detect_motif33(matrix)
    # motif_matrix = detect_motif32(matrix)
    m = matrix + motif_matrix
    return m,G,time.time()-time_begin

def normalizaion_arctan(array):
    """反正切归一化，反正切函数的值域就是[-pi/2,pi/2]
       公式：反正切值 * (2 / pi)
       :return 值域[-1,1]，原始大于0的数被映射到[0,1]，小于0的数被映射到[-1,0]
       """
    return np.arctan(array) * (2 / np.pi)

def get_acceptance(newMatrix,u,v):
    # 计算pu<-v 考虑新网络的边权重
    kv = np.sum(newMatrix[v])
    ku = np.sum(newMatrix[u])
    if (log(1 + ku) + log(1 + kv) == 0) :
        return 1
    return log(1 + kv) / (log(1 + ku) + log(1 + kv))
    # return newMatrix[u][v] / ku

def normalizaion_Matrix(matrix):
    newMatrix = np.zeros_like(matrix,dtype=float)
    for i in range(len(matrix)):
        newMatrix[i] = matrix[i] / sum(matrix[i])
    return newMatrix

