# import featuresByYBZDJ
from deepforest import *
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
from sklearn import metrics
import numpy as np
from skmultilearn.problem_transform import *
import scipy.io as scio
from bert_featureprocess import dataprocess
import random
import math
import scipy.io as si
from sklearn.ensemble import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

train_data = scio.loadmat('dataset/iamp/saveddata_AAC_DPC_PAAC_CTD.mat')["samples"]
train_bert = dataprocess('dataset/iamp/iamp_1kmer_-1.json')
train_ca = scio.loadmat('dataset/iamp/data_CA.mat')["samples"]

train_data = np.concatenate((train_data, train_ca, train_bert),axis=1).astype(np.float32)
print(train_data.shape)

train_label = scio.loadmat('dataset/iamp/savedlabel.mat')
train_label = train_label["labels"].astype(int)
print(train_label.shape)

# class MultiReliefF(object):
#
#     def __init__(self, n_neighbors=10, n_features_to_keep=10, n_selected=10):
#         """
#         初始化实例化对象
#         :param n_neighbors: 最近邻个数
#         :param n_features_to_keep: 选取特征相关统计量最大的数量
#         """
#         self.feature_scores = None
#         self.top_features = None
#         self.tree = None
#         self.n_neighbors = n_neighbors
#         self.n_features_to_keep = n_features_to_keep
#         self.n_selected = n_selected
#
#     def fit(self, X, y):
#         """
#         计算特征的相关统计量的大小
#         :param X: 数据部分
#         :param y: 标签部分
#         :return: 返回特征相关统计量数值的列表
#         """
#         # 记录每个特征的相关统计量，并初始化为0
#         self.feature_scores = np.zeros(X.shape[1])
#         # 获得了KDTree类实例化对象，后面用这个对象获得每个随机样本的K个最近邻
#         self.tree = KDTree(X)
#         num = X.max(axis=0) - X.min(axis=0)
#
#         # 在样本数量范围内进行不重复的随机采样self.n_selected次
#         random_list = random.sample(range(0, X.shape[0]), self.n_selected)
#
#         for source_index in random_list:
#             w = np.zeros(X.shape[1])
#             # 当前采用的是单位权重计算公式。由于多标签中标签之间可能有相关性，所以不能简单的拿单标签的去计算。
#             # 也可以采用其他权重计算公式
#             weight = np.sum(y[source_index]) / y.shape[1]
#
#             # 由于是多标签数据集，所以需要对每一个标签进行传统意义上的ReliefF查询，再对查询出的结果进行加权。
#             for label_index in range(y.shape[1]):
#                 label_data = y[:, label_index]
#                 # 此时是标签下的每一个分类
#                 diff_a = np.zeros(X.shape[1])
#                 diff_b = np.zeros(X.shape[1])
#
#                 # 对每一个标签进行去重，根据这个标签拥有的类别数进行循环，找到随机样本在每一类中的K个最近邻
#                 for label in np.unique(label_data):
#                     # 通过np.where方法找到所有当前类别的样本的索引
#                     each_class_samples_index = np.where(label_data == label)[0]
#                     # 调用KDTree方法找到最近邻
#                     data = X[each_class_samples_index, :]
#                     distances, indices = self.tree.query(
#                         X[source_index].reshape(1, -1), k=self.n_neighbors + 1)
#                     # 此时indices是每个标签下每个类别中的K个近邻,因为自己离自己最近，所以要从1开始
#                     indices = indices[0][1:]
#                     # 本次实验所采用的数据集是连续类型的，所以要采用距离计算
#                     # 如果是离散类型，那就直接调np.equal方法
#                     if label == label_data[source_index]:
#                         diff_a = np.sum((X[indices] - X[source_index]) ** 2, axis=0) / num
#                     else:
#                         prob = len(each_class_samples_index) / X.shape[0]
#                         # 异类样本的相关统计量计算需要再乘以异类样本占所有样本的比例
#                         diff_b += prob * (np.sum((X[indices] - X[source_index]) ** 2, axis=0) / num)
#                 # 最后对每一个标签的计算结果进行加权，就得到了最终每个样本计算的最终的相关统计量
#                 w += weight * (diff_b - diff_a) / (self.n_neighbors * self.n_selected)
#             self.feature_scores += w
#
#         # 根据对象初始化时的值，返回靠前的一些特征组成的数据子集。
#         self.top_features = np.argsort(self.feature_scores)[::-1]
#         return X[:, self.top_features[:self.n_features_to_keep]]



# train_data = MultiReliefF(n_neighbors=10, n_features_to_keep=500, n_selected=100).fit(train_data, train_label)
# print(train_data.shape)

'''
skmultilearn/problem_transform/lp.py 更改
selector = SelectFromModel(estimator=GradientBoostingClassifier()).fit(X, self.transform(y))
X = selector.transform(X)
X = self._ensure_input_format(X, sparse_format='csr', enforce_sparse=True)
self.classifier.fit(self._ensure_input_format(X),self.transform(y))
return selector
'''
kfold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=7)
for train, test in kfold.split(train_data, train_label):
    model = CascadeForestClassifier(random_state=1, n_estimators=5, use_predictor=True, n_trees=300)
    model_sepcnn = LabelPowerset(model)
    selector = model_sepcnn.fit(train_data[train], train_label[train])
    results = model_sepcnn.predict_proba(selector.transform(train_data[test])).toarray()

    length = train_label[test].shape[0]
    label_num = train_label[test].shape[1]
    with open("dataset/iamp/label.txt", "a+") as f:
        for i in range(length):
            f.write(">\n")
            l1 = train_label[test][i]
            string1 = '['
            for j in range(label_num):
                string1 += str(int(l1[j]))
                string1 += ','
            string1 = string1[:-1]
            string1 += ']\n'
            f.write(string1)
    f.close()

    with open("dataset/iamp/result.txt", "a+") as f:
        for i in range(len(results)):
            f.write(">\n")
            string = "["
            for j in range(label_num):
                string += str(results[i][j])
                string += ','
            string = string.strip(",")
            string += "]"
            f.write(string)
            f.write("\n")
    f.close()









