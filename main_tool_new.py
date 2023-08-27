# coding=utf-8
import networkx as nx
import numpy as np
# import cupy as np
from scipy.sparse import csr_matrix
# from cupyx.scipy.sparse import csr_matrix
import Graph
import igraph as ig
from matplotlib import pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def matrix_point_multiply(a_matrix, a_vector):
    """
        返回 向量与矩阵的点乘（自定义）
        于2021年12月7日进行优化，利用了np.einsum，大大提高了执行效率
    :param a_matrix: 传入一个矩阵
    :param a_vector: 传入一个向量
    :return: 矩阵和向量的点乘
    """
    return np.einsum('ij,i->ij', a_matrix.T, a_vector).T


class GRAPH(object):
    """
        于2021年4月编写，于2021年12月添加注释，功能为生成合适的图
    """
    # G=nx.karate_club_graph()
    # G = Graph.my_graph_igraph()
    G_nx = []
    G_ig = []
    G = []
    G1 = nx.Graph()
    G1.add_edge(1, 2)
    G1.add_edge(1, 3)
    G1.add_edge(3, 2)
    G1.add_edge(4, 3)
    G1.add_edge(4, 5)
    G1.add_edge(4, 6)
    G1.add_edge(5, 7)

    def __init__(self, network_filename):
        self.G_nx, self.G_ig = Graph.my_graph(network_filename)
        self.G = self.G_nx

    @staticmethod
    def nx2ig(nx_graph):
        d = nx.to_pandas_edgelist(nx_graph).values
        d=d-1
        newG=ig.Graph(d)
        return newG


class DATA(object):
    """
    于2021年4月编写，12月优化并添加注释，功能为提供合适的数据（第一层）
    """
    N = 0  # 数据的维度（图的节点总个数）
    AP = []  # 详见公式
    FAP = []  # 详见公式
    ap_current_number = 1  # 记录当前计算的ap的值的个数
    # A = np.array([[]])
    # B = np.array([[]])
    A = csr_matrix([[]])  # 邻接矩阵，稀疏矩阵形式的
    B = csr_matrix([[]]) # 一步转移矩阵，稀疏矩阵形式的
    BT = B.T  # 一步转移矩阵的转置
    BP = []  # 详见公式
    PP = [[]]  # 详见公式
    pp_current_number = 1  # 记录当前计算的pp的值的个数

    def __init__(self, g):
        """
        初始化各参数
        :param g: networkx图
        """
        self.N = len(g)
        # tmp = np.asarray(nx.adjacency_matrix(g).todense())
        # print("tmp获取")
        # tmp = tmp.astype('float')
        # for i in range(len(tmp)):
        #     for j in range(len(tmp[i])):
        #         tmp[i][j] = float(tmp[i][j])
        #         if tmp[i][j] != 0:
        #             print(tmp[i][j])
        # print("tmp形式转换完毕")
        # self.A = csr_matrix(tmp)# 邻接矩阵A
        self.A = csr_matrix(np.array(nx.adjacency_matrix(g).todense()))# 邻接矩阵A
        # print("邻接矩阵获取完毕")
        # en = np.ones(self.N)  # 全为1的N维向量
        en = [1] * self.N
        du = self.A.dot(en)  # Du为每个结点的度
        # print(du)
        self.Du=du
        np.where(du == 0, 1, du)  # 若某节点度为0，使其为1
        self.BT = csr_matrix(self.A.T / du)  # 得到一步转移矩阵
        self.B = self.BT.T

        self.AP = [[] for _ in range(500)]
        self.AP[1] = self.B.toarray()

        self.BP = [[] for _ in range(500)]
        self.__compute_bp()

        self.FAP = [[] for _ in range(500)]
        self.FAP[1] = self.AP[1]

        self.PP = [[] for _ in range(500)]
        self.PP[1] = self.AP[1]

    def __compute_bp(self):
        """
        计算BP的值
        """
        self.BP[self.ap_current_number] = self.AP[self.ap_current_number].diagonal()

    def __compute_ap(self, n):
        """
        计算未计算过的AP的值
        """
        the_last = self.ap_current_number
        for _ in range(n - the_last):
            self.AP[self.ap_current_number + 1] = self.BT.dot(self.AP[self.ap_current_number].T).T
            self.ap_current_number += 1
            self.__compute_bp()

    def __compute_fap(self, n):
        """
        计算未计算过的FAP的值
        :param n:最大步长
        """
        the_last = self.pp_current_number
        for j in range(the_last + 1, n + 1):
            ap = self.get_ap(j)
            temp = np.array([[0 for _ in range(self.N)] for _ in range(self.N)])
            for k in range(1, j - 1):
                temp = matrix_point_multiply(self.FAP[k], self.BP[j - k]) + temp
            self.FAP[j] = ap - temp
        del temp

    def __compute_pp(self, n):
        """
        计算未经计算的FAP和PP的值
        """
        the_last = self.pp_current_number
        for i in range(the_last + 1, n + 1):  # type: int
            last_pp1 = self.PP[i - 1]
            fap = self.__get_fap(i)  # 这里不能用self.FAP代替，因为还未计算
            self.pp_current_number += 1
            self.PP[i] = last_pp1 + fap

    def get_a(self):
        """
        :return: A的矩阵形式
        """
        return self.A.toarray()

    def get_b(self):
        """
        :return: B的矩阵形式
        """
        return self.B.toarray()

    def get_ap(self, n):
        if n > self.ap_current_number:
            self.__compute_ap(n)
        return self.AP[n]

    def get_bp(self, n):
        if n > self.ap_current_number:
            self.__compute_ap(n)
        return self.BP[n]

    def __get_fap(self, n):
        """
        对外封闭
        :param n:
        :return:
        """
        if n > self.pp_current_number:
            self.__compute_fap(n)
        return self.FAP[n]

    def get_pp(self, n):
        if n > self.pp_current_number:
            self.__compute_pp(n)
        return self.PP[n]

    def get_mp(self, n):
        if n > self.ap_current_number:
            self.__compute_ap(n)
        return self.AP[n].T*self.AP[n]

    def get_tp(self, n):
        if n > self.pp_current_number:
            self.__compute_pp(n)
        return self.PP[n].T*self.PP[n]

class GRModel:
    def __init__(self,nx_graph):
        nx_G =nx_graph
        self.N=len(nx_G)

    def subgraph(self,G):
        """
        :param G:一个图
        :return:该图的连通分量的集合，类型为list
        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_nodes_from([1,2,3])
        >>> G.add_edge(1,3)
        >>> subgraph(G)
        [[1,3],[2]]
        """
        list1 = []
        for c in nx.connected_components(G):
            list1.append(list(c))
        return list1

    def get_reconstructed_graph(self,resourceMatrix,num_edge=1):
        G1 = nx.Graph()
        for i in range(self.N):
            max_key= [0]*num_edge
            max_value= [0]*num_edge
            temp_value = resourceMatrix[i][i]
            resourceMatrix[i][i] = 0
            for j in range(num_edge):
                max_key[j] = np.argmax(resourceMatrix[i])
                max_value[j] = resourceMatrix[i][max_key[j]]
                resourceMatrix[i][max_key[j]] = 0
            for j in range(num_edge):
                resourceMatrix[i][max_key[j]] = max_value[j]
                G1.add_edge(i, max_key[j])
            resourceMatrix[i][i] = temp_value
        return self.subgraph(G1)


if __name__ == '__main__':

    # Gg,ii=Graph.my_graph("dolphins.txt")
    # data1=DATA(Gg)
    # pp3=data1.get_pp(3)
    # print("")

    # graph = GRAPH("C:/Users/Administrator/Desktop/LFR/LFR 5000 20 50 20 200/8/network2.dat")
    # graph=GRAPH("Email-Enron.txt")
    #
    # data=DATA(graph.G_nx)
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G.add_edge(3, 5)
    G.add_edge(3, 6)
    G.add_edge(4, 5)
    G.add_edge(4, 6)
    G.add_edge(5, 6)
    G.add_edge(5, 7)
    G.add_edge(6, 7)

    data=DATA(G)
    # data.get_ap(100)

    pp3=data.get_pp(3)
    model=GRModel(G)

    newG=model.get_reconstructed_graph(pp3,3)
    plt.axis("off")
    # nx.draw_networkx(G,nx.spring_layout(G),node_size=500,font_size=10)
    nx.draw_networkx(newG,nx.spring_layout(newG),node_size=500,font_size=10)
    plt.show()  # 展示图
    #
    # data = DATA(G)
    # ssss=data.get_ap(3)
    # sss = data.get_mp(1)
    print("")
    # data1 = main_tool.DATA(graph.G_nx)
    # a21 = data.get_ap(5)
    # a22 = data1.get_AP(5)
    # aa = data.get_pp(4)
    # aa1 = data1.get_PP(4)
    # e = [1 for i in range(data.N)]

    print("")
# coding=utf-8
# import networkx as nx
# import numpy as np
# # from scipy.sparse import csr_matrix
# import Graph
# import igraph as ig
# # from matplotlib import pyplot as plt
# import cupy as cp
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#
# def matrix_point_multiply(a_matrix, a_vector):
#     """
#         返回 向量与矩阵的点乘（自定义）
#         于2021年12月7日进行优化，利用了np.einsum，大大提高了执行效率
#     :param a_matrix: 传入一个矩阵
#     :param a_vector: 传入一个向量
#     :return: 矩阵和向量的点乘
#     """
#     return cp.einsum('ij,i->ij', a_matrix.T, a_vector).T
#
#
# class GRAPH(object):
#     """
#         于2021年4月编写，于2021年12月添加注释，功能为生成合适的图
#     """
#     # G=nx.karate_club_graph()
#     # G = Graph.my_graph_igraph()
#     G_nx = []
#     G_ig = []
#     G = []
#     G1 = nx.Graph()
#     G1.add_edge(1, 2)
#     G1.add_edge(1, 3)
#     G1.add_edge(3, 2)
#     G1.add_edge(4, 3)
#     G1.add_edge(4, 5)
#     G1.add_edge(4, 6)
#     G1.add_edge(5, 7)
#
#     def __init__(self, network_filename):
#         self.G_nx, self.G_ig = Graph.my_graph(network_filename)
#         self.G = self.G_nx
#
#     @staticmethod
#     def nx2ig(nx_graph):
#         d = nx.to_pandas_edgelist(nx_graph).values
#         d = d - 1
#         newG = ig.Graph(d)
#         return newG
#
#
# # def matrix_dot(a,b):
# #     return a.dot(b).T
#
#
# class DATA(object):
#     """
#     于2021年4月编写，12月优化并添加注释，功能为提供合适的数据（第一层）
#     """
#     N = 0  # 数据的维度（图的节点总个数）
#     AP = []  # 详见公式
#     FAP = []  # 详见公式
#     ap_current_number = 1  # 记录当前计算的ap的值的个数
#     A = cp.array([[]])  # 邻接矩阵，稀疏矩阵形式的
#     B = cp.array([[]])  # 一步转移矩阵，稀疏矩阵形式的
#     BT = B.T  # 一步转移矩阵的转置
#     BP = []  # 详见公式
#     PP = [[]]  # 详见公式
#     pp_current_number = 1  # 记录当前计算的pp的值的个数
#
#     def __init__(self, g):
#         """
#         初始化各参数
#         :param g: networkx图
#         """
#         self.N = len(g)
#         self.A = cp.array(nx.adjacency_matrix(g).todense())  # 邻接矩阵A
#         en = cp.array([1] * self.N)  # 全为1的N维向量
#         du = self.A.dot(en)  # Du为每个结点的度
#         self.Du = du
#         cp.where(du == 0, 1, du)  # 若某节点度为0，使其为1
#         self.BT = self.A.T / du  # 得到一步转移矩阵
#         self.B = self.BT.T
#
#         self.AP = [[] for _ in range(500)]
#         self.AP[1] = self.B
#
#         self.BP = [[] for _ in range(500)]
#         self.__compute_bp()
#
#         self.FAP = [[] for _ in range(500)]
#         self.FAP[1] = self.AP[1]
#
#         self.PP = [[] for _ in range(500)]
#         self.PP[1] = self.AP[1]
#
#     def __compute_bp(self):
#         """
#         计算BP的值
#         """
#         self.BP[self.ap_current_number] = self.AP[self.ap_current_number].diagonal()
#
#     def __compute_ap(self, n):
#         """
#         计算未计算过的AP的值
#         """
#         the_last = self.ap_current_number
#         for _ in range(n - the_last):
#             # self.AP[self.ap_current_number + 1] = matrix_dot(self.BT,self.AP[self.ap_current_number].T)
#             self.AP[self.ap_current_number + 1] = self.BT.dot(self.AP[self.ap_current_number].T).T
#             self.ap_current_number += 1
#             self.__compute_bp()
#
#     def __compute_fap(self, n):
#         """
#         计算未计算过的FAP的值
#         :param n:最大步长
#         """
#         the_last = self.pp_current_number
#         for j in range(the_last + 1, n + 1):
#             ap = self.get_ap(j)
#             temp = cp.array([[0 for _ in range(self.N)] for _ in range(self.N)])
#             for k in range(1, j - 1):
#                 temp = matrix_point_multiply(self.FAP[k], self.BP[j - k]) + temp
#             self.FAP[j] = ap - temp
#
#     def __compute_pp(self, n):
#         """
#         计算未经计算的FAP和PP的值
#         """
#         the_last = self.pp_current_number
#         for i in range(the_last + 1, n + 1):  # type: int
#             last_pp1 = self.PP[i - 1]
#             fap = self.__get_fap(i)  # 这里不能用self.FAP代替，因为还未计算
#             self.pp_current_number += 1
#             self.PP[i] = last_pp1 + fap
#
#     def get_a(self):
#         """
#         :return: A的矩阵形式
#         """
#         return self.A.toarray()
#
#     def get_b(self):
#         """
#         :return: B的矩阵形式
#         """
#         return self.B.toarray()
#
#     def get_ap(self, n):
#         if n > self.ap_current_number:
#             self.__compute_ap(n)
#         return self.AP[n]
#
#     def get_bp(self, n):
#         if n > self.ap_current_number:
#             self.__compute_ap(n)
#         return self.BP[n]
#
#     def __get_fap(self, n):
#         """
#         对外封闭
#         :param n:
#         :return:
#         """
#         if n > self.pp_current_number:
#             self.__compute_fap(n)
#         return self.FAP[n]
#
#     def get_pp(self, n):
#         if n > self.pp_current_number:
#             self.__compute_pp(n)
#         return self.PP[n]
#
#     def get_mp(self, n):
#         if n > self.ap_current_number:
#             self.__compute_ap(n)
#         return self.AP[n].T * self.AP[n]
#
#     def get_tp(self, n):
#         if n > self.pp_current_number:
#             self.__compute_pp(n)
#         return self.PP[n].T * self.PP[n]
#
#
# class GRModel:
#     def __init__(self, nx_graph):
#         nx_G = nx_graph
#         self.N = len(nx_G)
#
#     def get_reconstructed_graph(self, resourceMatrix, num_edge=1):
#         G1 = nx.Graph()
#         for i in range(self.N):
#             max_key = [0] * num_edge
#             max_value = [0] * num_edge
#             for j in range(num_edge):
#                 max_key[j] = cp.argmax(resourceMatrix[i])
#                 max_value[j] = resourceMatrix[i][max_key[j]]
#                 resourceMatrix[i][max_key[j]] = 0
#             for j in range(num_edge):
#                 resourceMatrix[i][max_key[j]] = max_value[j]
#                 G1.add_edge(i + 1, max_key[j] + 1)
#         return G1
#
#
# if __name__ == '__main__':
#     # Gg,ii=Graph.my_graph("dolphins.txt")
#     # data1=DATA(Gg)
#     # pp3=data1.get_pp(3)
#     # print("")
#
#     # graph = GRAPH("C:/Users/Administrator/Desktop/LFR/LFR 5000 20 50 20 200/8/network2.dat")
#     # graph=GRAPH("Email-Enron.txt")
#     #
#     # data=DATA(graph.G_nx)
#     G = nx.Graph()
#     G.add_edge(1, 2)
#     G.add_edge(1, 3)
#     G.add_edge(2, 3)
#     G.add_edge(3, 4)
#     G.add_edge(3, 5)
#     G.add_edge(3, 6)
#     G.add_edge(4, 5)
#     G.add_edge(4, 6)
#     G.add_edge(5, 6)
#     G.add_edge(5, 7)
#     G.add_edge(6, 7)
#
#     data = DATA(G)
#     # data.get_ap(100)
#
#     pp3 = data.get_pp(3)
#
#     model = GRModel(G)
#
#     newG = model.get_reconstructed_graph(pp3, 3)
#     plt.axis("off")
#     # nx.draw_networkx(G,nx.spring_layout(G),node_size=500,font_size=10)
#     nx.draw_networkx(newG, nx.spring_layout(newG), node_size=500, font_size=10)
#     plt.show()  # 展示图
#     #
#     # data = DATA(G)
#     # ssss=data.get_ap(3)
#     # sss = data.get_mp(1)
#     print("")
#     # data1 = main_tool.DATA(graph.G_nx)
#     # a21 = data.get_ap(5)
#     # a22 = data1.get_AP(5)
#     # aa = data.get_pp(4)
#     # aa1 = data1.get_PP(4)
#     # e = [1 for i in range(data.N)]
#
#     print("")
