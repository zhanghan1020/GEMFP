from sklearn import metrics
from sklearn.cluster import KMeans

from matrix import *
import copy

def cal_Q(G,k):  # 计算Q
    result = []
    for j in range(len(G.nodes)):
        result.append([])
    index = 0
    for node in G.nodes:
        result[index].append(G.nodes[node]['embed'])
        index = index + 1
    result = np.array(result).reshape(len(G.nodes), maxd)

    model = KMeans(n_clusters=k)  # 分为k类
    model.fit(result)
    label = model.labels_
    #转换格式
    Gnodes = list(G.nodes())
    tmp = dict()
    for i in range(len(label)):
        if label[i] not in tmp.keys():
            com = []
            com.append(Gnodes[i])
            tmp[label[i]] = com
        else:tmp[label[i]].append(Gnodes[i])
    partition = list(tmp.values())
    m = len(G.edges(None, False))  # 如果为真，则返回3元组（u、v、ddict）中的边缘属性dict。如果为false，则返回2元组（u，v）
    # print(G.edges(None,False))
    # print("=======6666666")
    a = []
    e = []
    for community in partition:  # 把每一个联通子图拿出来
        t = 0.0
        for node in community:  # 找出联通子图的每一个顶点
            t += len([x for x in G.neighbors(node)])  # G.neighbors(node)找node节点的邻接节点
        a.append(t / (2 * m))
    #             self.zidian[t/(2*m)]=community
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if (G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q

def prepare(G,filename):
    file_read = open(filename, "r")
    real_label1 = dict()
    real_label = []

    for line in file_read:
        line = line.split('\t')
        real_label1[int(line[0])] = int(line[len(line) - 1].split("\n")[0])-1

    test_data_1 = sorted(real_label1.items(), key=lambda x: x[0])
    for i in range(len(test_data_1)):
        real_label.append(test_data_1[i][1])
    real_label = np.array(real_label,dtype=int)

    k = max(real_label)+1 # 社团数量
    print("社团数量："+str(k))

    result = []
    for node in G.nodes:
        result.append(G.nodes[node]['embed'])
    result = np.array(result).reshape(len(G.nodes), maxd)

    print("开始聚类")
    model = KMeans(n_clusters=k)  # 分为k类
    model.fit(result)
    label = model.labels_
    return real_label,label,k

def Feature_Propagation(G,m,threshold):
    # 开始特征传播
    epochs = 100  # 迭代次数
    flag = False
    time_begin = time.time()
    nodes = np.array(G.nodes())
    res = list()

    for node in G.nodes:
        res.append(G.nodes[node]['embed'])

    for epoch in range(epochs):
        thres = []
        for node in G.nodes:  # node为w的下标
            # 暂定用异步更新方式
            vector = copy.copy(res[node])
            for nei in G[node]:
                index1 = np.where(nodes == node)[0][0]
                index2 = np.where(nodes == nei)[0][0]
                vector += get_acceptance(m, index1, index2) * G.nodes[nei]['embed']  # 2 cora 0.4369287658923185

            vector = normalizaion_arctan(vector)

            thres.append(sum(abs(vector - res[node])) / 256)
            res[node] = copy.copy(vector)
            G.nodes[node]['embed'] = copy.copy(res[node])
            th = sum(thres) / len(thres)
            # th = np.sqrt(np.sum(np.square(vector-res[node])))

            if th < threshold:
                flag = True
        if flag:
            print("迭代了%d" % (epoch + 1) + "次，结束更新")
            break
        print("第" + str(epoch + 1) + "次迭代结束:"+str(th))
    del epoch, node

if __name__ == '__main__':

    m, G ,t= get_newMatrix("./Data/doph11.txt")
    print("特征矩阵计算完成")

    label_file = "./Data/doph11realcomm.txt"
    Feature_Propagation(G,m,0.001)

    real_label, label, k = prepare(G, label_file)
    print("开始计算nmi:")
    result_NMI = metrics.normalized_mutual_info_score(real_label, label)
    print("NMI:" + str(result_NMI))

    print("开始计算ARI：")
    result_ARI = metrics.adjusted_rand_score(real_label, label)
    print("ARI:" + str(result_ARI))

    print("开始计算Q:")
    result_q = cal_Q(G, k)
    print("Q:" + str(result_q))

