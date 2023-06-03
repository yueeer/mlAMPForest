from deepforest import *
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
from sklearn import metrics
import numpy as np
from skmultilearn.problem_transform import *
import scipy.io as scio
from skmultilearn.ensemble import *
from bert_featureprocess import dataprocess
# from myMLSMOTE import *
from sklearn.ensemble import *

from sklearn.feature_selection import *

train_data = scio.loadmat('dataset/iamp/saveddata_AAC_DPC_PAAC_CTD.mat')["samples"]
train_bert = dataprocess('dataset/iamp/iamp_1kmer_-1.json')
train_ca = scio.loadmat('dataset/iamp/data_CA.mat')["samples"]

train_data = np.concatenate((train_data, train_ca, train_bert),axis=1).astype(np.float32)
print(train_data.shape)

train_label = scio.loadmat('dataset/iamp/savedlabel.mat')
train_label = train_label["labels"].astype(int)
print(train_label.shape)


for index in range(train_label.shape[1]):
    label = train_label[:,index]
    data = train_data

    selector = SelectFromModel(estimator=GradientBoostingClassifier()).fit(train_data, label)
    data = selector.transform(train_data)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    true_labels = []
    predict_results = []
    for train, test in kfold.split(data, label):
        # X_sub, y_sub = get_minority_instace(data[train], label[train])  # Getting minority instance of that datframe
        # X_res, y_res = MLSMOTE(X_sub, y_sub, X_sub.shape[0]*5)  # Applying MLSMOTE to augment the dataframe
        # new_X = np.concatenate((data[train], X_res), axis=0)
        # new_y = np.concatenate((label[train], y_res), axis=0)

        model = CascadeForestClassifier(random_state=1, n_estimators=5, use_predictor=True, n_trees=300)
        model_sepcnn = model

        model_sepcnn.fit(data[train],label[train])
        results = model_sepcnn.predict(data[test])

        for i in label[test]:
            true_labels.append(i)
        for i in results:
            predict_results.append(i)

    length = len(true_labels)
    label_file = "dataset/iamp/label" + str(index) + ".txt"
    result_file = "dataset/iamp/result" + str(index) + ".txt"
    with open(label_file, "w") as f:
        for i in range(length):
            f.write(str(true_labels[i]) + "\n")
    f.close()

    with open(result_file, "w") as f:
        for i in range(length):
            f.write(str(predict_results[i]) + "\n")
    f.close()

