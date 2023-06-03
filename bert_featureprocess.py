#[NONE, 768]
import numpy as np
import json
def dataprocess(filename):
    with open(filename, 'r') as f:
        papers = []
        for line in f.readlines():  # 第line条序列
            dic = json.loads(line)
            features = dic["features"]
            temp = features[0]
            layers = temp["layers"][0]
            value = layers["values"]
            # print(len(value))
            papers.append(value)
    f.close()
    papers = np.array(papers)
    return papers


