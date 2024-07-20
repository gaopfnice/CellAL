# -*- coding: utf-8 -*-
import time
import numpy as np
from lightgbm.sklearn import LGBMClassifier
from sklearn.decomposition import PCA
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_importance
from matplotlib import pyplot as plt

def Sample_generation(seed,path):
    positive_feature = []
    negative_feature = []
    ligand_dict = {}
    receptor_dict = {}
    name_L = []
    name_R = []
    LRI_name = []
    feature = []
    LRI_name_z = []
    negative_sample_feature = []


    L1 = pd.read_csv(path+"l1.csv", header=None, index_col=None).to_numpy()
    L2 = pd.read_csv(path+"l2.csv", header=None, index_col=None).to_numpy()
    L3 = pd.read_csv(path+"l3.csv", header=None, index_col=None).to_numpy()
    L4 = pd.read_csv(path+"l4.csv", header=None, index_col=None).to_numpy()
    ligand_feature = np.hstack((L1, L2, L3, L4))

    R1 = pd.read_csv(path + "r1.csv", header=None, index_col=None).to_numpy()
    R2 = pd.read_csv(path + "r2.csv", header=None, index_col=None).to_numpy()
    R3 = pd.read_csv(path + "r3.csv", header=None, index_col=None).to_numpy()
    R4 = pd.read_csv(path + "r4.csv", header=None, index_col=None).to_numpy()
    receptor_feature = np.hstack((R1, R2, R3, R4))



    interaction = pd.read_csv(path+"interaction.csv", header=0, index_col=0).to_numpy()
    df = pd.read_csv(path + 'interaction.csv', header=0, index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])
    for i, name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]

    with open(path + 'receptor.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
    for i, name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    count0 = 0
    count1 = 0
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            temp = np.append(ligand_feature[i], receptor_feature[j])  # 加行和列的标签
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)
                LRI_name.append(df.index[i] + ' ' + df.columns[j])
                count1 += 1
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)
                LRI_name_z.append(df.index[i] + ' ' + df.columns[j])#LRI_name_z所有负样本的名字
                count0 += 1
    label00 = np.zeros((len(negative_feature), 1))#所有负样本添加标签

    LRI_known = LRI_name
    LRI_known = pd.DataFrame(LRI_known)
    LRI_known.to_csv(path + "LRI_name_known.csv", header=None, index=None)#保存已知正样本name

    # negative_feature = np.hstack([negative_feature, label00])
    # np.savetxt(path + "negative_feature.txt", negative_feature, delimiter=',')#保存所有负样本
    # LRI_name_z = pd.DataFrame(LRI_name_z)
    # LRI_name_z.to_csv(path + 'LRI_name_z.csv', header=None, index=None)#保存所有负样本name
    row = []
    col = []
    [row0, column0] = np.where(df.values == 0)

    rand = np.random.RandomState(seed)
    num = rand.randint(row0.shape, size=count1)
    for i in num:
        row.append(row0[i])
        col.append(column0[i])

    for i, j in zip(row, col):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = np.append(lig_tri_fea, rec_tri_fea)
        LRI_name.append(df.index[i] + ' ' + df.columns[j])
        negative_sample_feature.append(temp_f)

    #LRI_name = pd.DataFrame(LRI_name)
    #LRI_name.to_csv(path + 'LRI_name.csv', header=None, index=None)#同等数量0，1的数据集
    feature = np.vstack((positive_feature, negative_sample_feature))



    label1 = np.ones((len(positive_feature), 1))
    label0 = np.zeros((len(negative_sample_feature), 1))
    label = np.vstack((label1, label0))

    # XGBoost特征选择（降维）
    xgb_model = xgb.XGBClassifier()
    xgbresult1 = xgb_model.fit(feature, label.ravel())  # python 中的 ravel() 函数将数组多维度拉成一维数组
    feature_importance = xgbresult1.feature_importances_
    feature_number = -feature_importance
    H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    mask = H1[:400]
    train_data = feature[:, mask]

    # #lightGBM特征选择（降维）
    # lgb_model = LGBMClassifier()
    # lgbresults = lgb_model.fit(feature, label.ravel())
    # feature_importance = lgbresults.feature_importances_
    # feature_number = -feature_importance
    # H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # mask = H1[:400]
    # train_data = feature[:, mask]


    # 标准化
    # std = StandardScaler()
    # train_data = std.fit_transform(train_data)
    # # #归一化
    # mm = MinMaxScaler()
    # train_data = mm.fit_transform(train_data)

    feature = np.hstack((train_data, label))
    # np.savetxt(path+"train.txt", feature, delimiter=',')
    feature = pd.DataFrame(feature)
    feature.to_csv(path + 'train.csv', header=None, index=None)
    #train表示训练用的数据集，其中包括正样本，负样本。test表示预测用的数据集，其中全部都是负样本，预测所有的0中哪些可能是1
    return feature


def test_generation(seed, path):
    positive_feature = []
    negative_feature = []
    ligand_dict = {}
    receptor_dict = {}
    name_L = []
    name_R = []
    LRI_name = []
    feature = []
    LRI_name_z = []
    negative_sample_feature = []

    L1 = pd.read_csv(path + "l1.csv", header=None, index_col=None).to_numpy()
    L2 = pd.read_csv(path + "l2.csv", header=None, index_col=None).to_numpy()
    L3 = pd.read_csv(path + "l3.csv", header=None, index_col=None).to_numpy()
    L4 = pd.read_csv(path + "l4.csv", header=None, index_col=None).to_numpy()
    ligand_feature = np.hstack((L1, L2, L3, L4))
    print(ligand_feature.shape)
    R1 = pd.read_csv(path + "r1.csv", header=None, index_col=None).to_numpy()
    R2 = pd.read_csv(path + "r2.csv", header=None, index_col=None).to_numpy()
    R3 = pd.read_csv(path + "r3.csv", header=None, index_col=None).to_numpy()
    R4 = pd.read_csv(path + "r4.csv", header=None, index_col=None).to_numpy()
    receptor_feature = np.hstack((R1, R2, R3, R4))
    print(receptor_feature.shape)

    interaction = pd.read_csv(path + "interaction.csv", header=0, index_col=0).to_numpy()
    df = pd.read_csv(path + 'interaction.csv', header=0, index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])

    for i, name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]

    with open(path + 'receptor.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
    for i, name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    count0 = 0
    count1 = 0

    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            temp = np.append(ligand_feature[i], receptor_feature[j])  # 加行和列的标签
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)
                LRI_name.append(df.index[i] + ' ' + df.columns[j])
                count1 += 1
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)
                LRI_name_z.append(df.index[i] + ' ' + df.columns[j])
                count0 += 1

    label00 = np.zeros((len(negative_feature), 1))  # 所有负样本添加标签
    LRI_name_z = pd.DataFrame(LRI_name_z)
    LRI_name_z.to_csv(path + 'LRI_name_z.csv', header=None, index=None)  # 保存所有负样本name

    #lightGBM特征选择（降维）
    lgb_model = LGBMClassifier()
    lgbresults = lgb_model.fit(negative_feature, label00.ravel())
    feature_importance = lgbresults.feature_importances_
    feature_number = -feature_importance
    H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    mask = H1[:400]
    negative_feature = np.array(negative_feature)
    test_data = negative_feature[:, mask]
    test_data = pd.DataFrame(test_data)
    test_data.to_csv(path + "test.csv" ,header=None, index=None)
    # np.savetxt(path + "negative_feature.txt", test_data, delimiter=',')  # 保存所有负样本

    return test_data

def Sample_generation200(seed,path):
    positive_feature = []
    negative_feature = []
    ligand_dict = {}
    receptor_dict = {}
    name_L = []
    name_R = []
    LRI_name = []
    feature = []
    LRI_name_z = []
    negative_sample_feature = []


    L1 = pd.read_csv(path+"l1.csv", header=None, index_col=None).to_numpy()
    L2 = pd.read_csv(path+"l2.csv", header=None, index_col=None).to_numpy()
    L3 = pd.read_csv(path+"l3.csv", header=None, index_col=None).to_numpy()
    L4 = pd.read_csv(path+"l4.csv", header=None, index_col=None).to_numpy()
    ligand_feature = np.hstack((L1, L2, L3, L4))

    R1 = pd.read_csv(path + "r1.csv", header=None, index_col=None).to_numpy()
    R2 = pd.read_csv(path + "r2.csv", header=None, index_col=None).to_numpy()
    R3 = pd.read_csv(path + "r3.csv", header=None, index_col=None).to_numpy()
    R4 = pd.read_csv(path + "r4.csv", header=None, index_col=None).to_numpy()
    receptor_feature = np.hstack((R1, R2, R3, R4))



    interaction = pd.read_csv(path+"interaction.csv", header=0, index_col=0).to_numpy()
    df = pd.read_csv(path + 'interaction.csv', header=0, index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])
    for i, name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]

    with open(path + 'receptor.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
    for i, name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    count0 = 0
    count1 = 0
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            temp = np.append(ligand_feature[i], receptor_feature[j])  # 加行和列的标签
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)
                LRI_name.append(df.index[i] + ' ' + df.columns[j])
                count1 += 1
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)
                LRI_name_z.append(df.index[i] + ' ' + df.columns[j])#LRI_name_z所有负样本的名字
                count0 += 1
    label00 = np.zeros((len(negative_feature), 1))#所有负样本添加标签

    LRI_known = LRI_name
    LRI_known = pd.DataFrame(LRI_known)
    LRI_known.to_csv(path + "LRI_name_known.csv", header=None, index=None)#保存已知正样本name

    # negative_feature = np.hstack([negative_feature, label00])
    # np.savetxt(path + "negative_feature.txt", negative_feature, delimiter=',')#保存所有负样本
    # LRI_name_z = pd.DataFrame(LRI_name_z)
    # LRI_name_z.to_csv(path + 'LRI_name_z.csv', header=None, index=None)#保存所有负样本name
    row = []
    col = []
    [row0, column0] = np.where(df.values == 0)

    rand = np.random.RandomState(seed)
    num = rand.randint(row0.shape, size=count1)
    for i in num:
        row.append(row0[i])
        col.append(column0[i])

    for i, j in zip(row, col):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = np.append(lig_tri_fea, rec_tri_fea)
        LRI_name.append(df.index[i] + ' ' + df.columns[j])
        negative_sample_feature.append(temp_f)

    #LRI_name = pd.DataFrame(LRI_name)
    #LRI_name.to_csv(path + 'LRI_name.csv', header=None, index=None)#同等数量0，1的数据集
    feature = np.vstack((positive_feature, negative_sample_feature))



    label1 = np.ones((len(positive_feature), 1))
    label0 = np.zeros((len(negative_sample_feature), 1))
    label = np.vstack((label1, label0))

    #XGBoost特征选择（降维）
    xgb_model = xgb.XGBClassifier()
    xgbresult1 = xgb_model.fit(feature, label.ravel())  # python 中的 ravel() 函数将数组多维度拉成一维数组
    feature_importance = xgbresult1.feature_importances_
    feature_number = -feature_importance
    H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    mask = H1[:200]
    train_data = feature[:, mask]

    #lightGBM特征选择（降维）
    # lgb_model = LGBMClassifier()
    # lgbresults = lgb_model.fit(feature, label.ravel())
    # feature_importance = lgbresults.feature_importances_
    # feature_number = -feature_importance
    # H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # mask = H1[:200]
    # train_data = feature[:, mask]


    # 标准化
    # std = StandardScaler()
    # train_data = std.fit_transform(train_data)
    # # #归一化
    # mm = MinMaxScaler()
    # train_data = mm.fit_transform(train_data)

    feature = np.hstack((train_data, label))
    # np.savetxt(path+"train.txt", feature, delimiter=',')
    feature = pd.DataFrame(feature)
    feature.to_csv(path + 'train.csv', header=None, index=None)
    #train表示训练用的数据集，其中包括正样本，负样本。test表示预测用的数据集，其中全部都是负样本，预测所有的0中哪些可能是1
    return feature


def Sample_generation300(seed,path):
    positive_feature = []
    negative_feature = []
    ligand_dict = {}
    receptor_dict = {}
    name_L = []
    name_R = []
    LRI_name = []
    feature = []
    LRI_name_z = []
    negative_sample_feature = []


    L1 = pd.read_csv(path+"l1.csv", header=None, index_col=None).to_numpy()
    L2 = pd.read_csv(path+"l2.csv", header=None, index_col=None).to_numpy()
    L3 = pd.read_csv(path+"l3.csv", header=None, index_col=None).to_numpy()
    L4 = pd.read_csv(path+"l4.csv", header=None, index_col=None).to_numpy()
    ligand_feature = np.hstack((L1, L2, L3, L4))

    R1 = pd.read_csv(path + "r1.csv", header=None, index_col=None).to_numpy()
    R2 = pd.read_csv(path + "r2.csv", header=None, index_col=None).to_numpy()
    R3 = pd.read_csv(path + "r3.csv", header=None, index_col=None).to_numpy()
    R4 = pd.read_csv(path + "r4.csv", header=None, index_col=None).to_numpy()
    receptor_feature = np.hstack((R1, R2, R3, R4))



    interaction = pd.read_csv(path+"interaction.csv", header=0, index_col=0).to_numpy()
    df = pd.read_csv(path + 'interaction.csv', header=0, index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])
    for i, name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]

    with open(path + 'receptor.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
    for i, name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    count0 = 0
    count1 = 0
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            temp = np.append(ligand_feature[i], receptor_feature[j])  # 加行和列的标签
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)
                LRI_name.append(df.index[i] + ' ' + df.columns[j])
                count1 += 1
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)
                LRI_name_z.append(df.index[i] + ' ' + df.columns[j])#LRI_name_z所有负样本的名字
                count0 += 1
    label00 = np.zeros((len(negative_feature), 1))#所有负样本添加标签

    LRI_known = LRI_name
    LRI_known = pd.DataFrame(LRI_known)
    LRI_known.to_csv(path + "LRI_name_known.csv", header=None, index=None)#保存已知正样本name

    # negative_feature = np.hstack([negative_feature, label00])
    # np.savetxt(path + "negative_feature.txt", negative_feature, delimiter=',')#保存所有负样本
    # LRI_name_z = pd.DataFrame(LRI_name_z)
    # LRI_name_z.to_csv(path + 'LRI_name_z.csv', header=None, index=None)#保存所有负样本name
    row = []
    col = []
    [row0, column0] = np.where(df.values == 0)

    rand = np.random.RandomState(seed)
    num = rand.randint(row0.shape, size=count1)
    for i in num:
        row.append(row0[i])
        col.append(column0[i])

    for i, j in zip(row, col):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = np.append(lig_tri_fea, rec_tri_fea)
        LRI_name.append(df.index[i] + ' ' + df.columns[j])
        negative_sample_feature.append(temp_f)

    #LRI_name = pd.DataFrame(LRI_name)
    #LRI_name.to_csv(path + 'LRI_name.csv', header=None, index=None)#同等数量0，1的数据集
    feature = np.vstack((positive_feature, negative_sample_feature))



    label1 = np.ones((len(positive_feature), 1))
    label0 = np.zeros((len(negative_sample_feature), 1))
    label = np.vstack((label1, label0))

    #XGBoost特征选择（降维）
    xgb_model = xgb.XGBClassifier()
    xgbresult1 = xgb_model.fit(feature, label.ravel())  # python 中的 ravel() 函数将数组多维度拉成一维数组
    feature_importance = xgbresult1.feature_importances_
    feature_number = -feature_importance
    H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    mask = H1[:300]
    train_data = feature[:, mask]

    #lightGBM特征选择（降维）
    # lgb_model = LGBMClassifier()
    # lgbresults = lgb_model.fit(feature, label.ravel())
    # feature_importance = lgbresults.feature_importances_
    # feature_number = -feature_importance
    # H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # mask = H1[:300]
    # train_data = feature[:, mask]


    # 标准化
    # std = StandardScaler()
    # train_data = std.fit_transform(train_data)
    # # #归一化
    # mm = MinMaxScaler()
    # train_data = mm.fit_transform(train_data)

    feature = np.hstack((train_data, label))
    # np.savetxt(path+"train.txt", feature, delimiter=',')
    feature = pd.DataFrame(feature)
    feature.to_csv(path + 'train.csv', header=None, index=None)
    #train表示训练用的数据集，其中包括正样本，负样本。test表示预测用的数据集，其中全部都是负样本，预测所有的0中哪些可能是1
    return feature

def Sample_generation350(seed,path):
    positive_feature = []
    negative_feature = []
    ligand_dict = {}
    receptor_dict = {}
    name_L = []
    name_R = []
    LRI_name = []
    feature = []
    LRI_name_z = []
    negative_sample_feature = []


    L1 = pd.read_csv(path+"l1.csv", header=None, index_col=None).to_numpy()
    L2 = pd.read_csv(path+"l2.csv", header=None, index_col=None).to_numpy()
    L3 = pd.read_csv(path+"l3.csv", header=None, index_col=None).to_numpy()
    L4 = pd.read_csv(path+"l4.csv", header=None, index_col=None).to_numpy()
    ligand_feature = np.hstack((L1, L2, L3, L4))

    R1 = pd.read_csv(path + "r1.csv", header=None, index_col=None).to_numpy()
    R2 = pd.read_csv(path + "r2.csv", header=None, index_col=None).to_numpy()
    R3 = pd.read_csv(path + "r3.csv", header=None, index_col=None).to_numpy()
    R4 = pd.read_csv(path + "r4.csv", header=None, index_col=None).to_numpy()
    receptor_feature = np.hstack((R1, R2, R3, R4))



    interaction = pd.read_csv(path+"interaction.csv", header=0, index_col=0).to_numpy()
    df = pd.read_csv(path + 'interaction.csv', header=0, index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])
    for i, name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]

    with open(path + 'receptor.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
    for i, name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    count0 = 0
    count1 = 0
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            temp = np.append(ligand_feature[i], receptor_feature[j])  # 加行和列的标签
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)
                LRI_name.append(df.index[i] + ' ' + df.columns[j])
                count1 += 1
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)
                LRI_name_z.append(df.index[i] + ' ' + df.columns[j])#LRI_name_z所有负样本的名字
                count0 += 1
    label00 = np.zeros((len(negative_feature), 1))#所有负样本添加标签

    LRI_known = LRI_name
    LRI_known = pd.DataFrame(LRI_known)
    LRI_known.to_csv(path + "LRI_name_known.csv", header=None, index=None)#保存已知正样本name

    # negative_feature = np.hstack([negative_feature, label00])
    # np.savetxt(path + "negative_feature.txt", negative_feature, delimiter=',')#保存所有负样本
    # LRI_name_z = pd.DataFrame(LRI_name_z)
    # LRI_name_z.to_csv(path + 'LRI_name_z.csv', header=None, index=None)#保存所有负样本name
    row = []
    col = []
    [row0, column0] = np.where(df.values == 0)

    rand = np.random.RandomState(seed)
    num = rand.randint(row0.shape, size=count1)
    for i in num:
        row.append(row0[i])
        col.append(column0[i])

    for i, j in zip(row, col):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = np.append(lig_tri_fea, rec_tri_fea)
        LRI_name.append(df.index[i] + ' ' + df.columns[j])
        negative_sample_feature.append(temp_f)

    #LRI_name = pd.DataFrame(LRI_name)
    #LRI_name.to_csv(path + 'LRI_name.csv', header=None, index=None)#同等数量0，1的数据集
    feature = np.vstack((positive_feature, negative_sample_feature))



    label1 = np.ones((len(positive_feature), 1))
    label0 = np.zeros((len(negative_sample_feature), 1))
    label = np.vstack((label1, label0))

    #XGBoost特征选择（降维）
    xgb_model = xgb.XGBClassifier()
    xgbresult1 = xgb_model.fit(feature, label.ravel())  # python 中的 ravel() 函数将数组多维度拉成一维数组
    feature_importance = xgbresult1.feature_importances_
    feature_number = -feature_importance
    H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    mask = H1[:350]
    train_data = feature[:, mask]

    #lightGBM特征选择（降维）
    # lgb_model = LGBMClassifier()
    # lgbresults = lgb_model.fit(feature, label.ravel())
    # feature_importance = lgbresults.feature_importances_
    # feature_number = -feature_importance
    # H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # mask = H1[:350]
    # train_data = feature[:, mask]


    # 标准化
    # std = StandardScaler()
    # train_data = std.fit_transform(train_data)
    # # #归一化
    # mm = MinMaxScaler()
    # train_data = mm.fit_transform(train_data)

    feature = np.hstack((train_data, label))
    # np.savetxt(path+"train.txt", feature, delimiter=',')
    feature = pd.DataFrame(feature)
    feature.to_csv(path + 'train.csv', header=None, index=None)
    #train表示训练用的数据集，其中包括正样本，负样本。test表示预测用的数据集，其中全部都是负样本，预测所有的0中哪些可能是1
    return feature


def Sample_generation450(seed, path):
    positive_feature = []
    negative_feature = []
    ligand_dict = {}
    receptor_dict = {}
    name_L = []
    name_R = []
    LRI_name = []
    feature = []
    LRI_name_z = []
    negative_sample_feature = []


    L1 = pd.read_csv(path+"l1.csv", header=None, index_col=None).to_numpy()
    L2 = pd.read_csv(path+"l2.csv", header=None, index_col=None).to_numpy()
    L3 = pd.read_csv(path+"l3.csv", header=None, index_col=None).to_numpy()
    L4 = pd.read_csv(path+"l4.csv", header=None, index_col=None).to_numpy()
    ligand_feature = np.hstack((L1, L2, L3, L4))

    R1 = pd.read_csv(path + "r1.csv", header=None, index_col=None).to_numpy()
    R2 = pd.read_csv(path + "r2.csv", header=None, index_col=None).to_numpy()
    R3 = pd.read_csv(path + "r3.csv", header=None, index_col=None).to_numpy()
    R4 = pd.read_csv(path + "r4.csv", header=None, index_col=None).to_numpy()
    receptor_feature = np.hstack((R1, R2, R3, R4))



    interaction = pd.read_csv(path+"interaction.csv", header=0, index_col=0).to_numpy()
    df = pd.read_csv(path + 'interaction.csv', header=0, index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])
    for i, name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]

    with open(path + 'receptor.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
    for i, name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    count0 = 0
    count1 = 0
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            temp = np.append(ligand_feature[i], receptor_feature[j])  # 加行和列的标签
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)
                LRI_name.append(df.index[i] + ' ' + df.columns[j])
                count1 += 1
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)
                LRI_name_z.append(df.index[i] + ' ' + df.columns[j])#LRI_name_z所有负样本的名字
                count0 += 1
    label00 = np.zeros((len(negative_feature), 1))#所有负样本添加标签

    LRI_known = LRI_name
    LRI_known = pd.DataFrame(LRI_known)
    LRI_known.to_csv(path + "LRI_name_known.csv", header=None, index=None)#保存已知正样本name

    # negative_feature = np.hstack([negative_feature, label00])
    # np.savetxt(path + "negative_feature.txt", negative_feature, delimiter=',')#保存所有负样本
    # LRI_name_z = pd.DataFrame(LRI_name_z)
    # LRI_name_z.to_csv(path + 'LRI_name_z.csv', header=None, index=None)#保存所有负样本name
    row = []
    col = []
    [row0, column0] = np.where(df.values == 0)

    rand = np.random.RandomState(seed)
    num = rand.randint(row0.shape, size=count1)
    for i in num:
        row.append(row0[i])
        col.append(column0[i])

    for i, j in zip(row, col):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = np.append(lig_tri_fea, rec_tri_fea)
        LRI_name.append(df.index[i] + ' ' + df.columns[j])
        negative_sample_feature.append(temp_f)

    #LRI_name = pd.DataFrame(LRI_name)
    #LRI_name.to_csv(path + 'LRI_name.csv', header=None, index=None)#同等数量0，1的数据集
    feature = np.vstack((positive_feature, negative_sample_feature))



    label1 = np.ones((len(positive_feature), 1))
    label0 = np.zeros((len(negative_sample_feature), 1))
    label = np.vstack((label1, label0))

    #XGBoost特征选择（降维）
    xgb_model = xgb.XGBClassifier()
    xgbresult1 = xgb_model.fit(feature, label.ravel())  # python 中的 ravel() 函数将数组多维度拉成一维数组
    feature_importance = xgbresult1.feature_importances_
    feature_number = -feature_importance
    H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    mask = H1[:450]
    train_data = feature[:, mask]

    #lightGBM特征选择（降维）
    # lgb_model = LGBMClassifier()
    # lgbresults = lgb_model.fit(feature, label.ravel())
    # feature_importance = lgbresults.feature_importances_
    # feature_number = -feature_importance
    # H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # mask = H1[:450]
    # train_data = feature[:, mask]


    # 标准化
    # std = StandardScaler()
    # train_data = std.fit_transform(train_data)
    # # #归一化
    # mm = MinMaxScaler()
    # train_data = mm.fit_transform(train_data)

    feature = np.hstack((train_data, label))
    # np.savetxt(path+"train.txt", feature, delimiter=',')
    feature = pd.DataFrame(feature)
    feature.to_csv(path + 'train.csv', header=None, index=None)
    #train表示训练用的数据集，其中包括正样本，负样本。test表示预测用的数据集，其中全部都是负样本，预测所有的0中哪些可能是1
    return feature


def Sample_generation500(seed,path):
    positive_feature = []
    negative_feature = []
    ligand_dict = {}
    receptor_dict = {}
    name_L = []
    name_R = []
    LRI_name = []
    feature = []
    LRI_name_z = []
    negative_sample_feature = []


    L1 = pd.read_csv(path+"l1.csv", header=None, index_col=None).to_numpy()
    L2 = pd.read_csv(path+"l2.csv", header=None, index_col=None).to_numpy()
    L3 = pd.read_csv(path+"l3.csv", header=None, index_col=None).to_numpy()
    L4 = pd.read_csv(path+"l4.csv", header=None, index_col=None).to_numpy()
    ligand_feature = np.hstack((L1, L2, L3, L4))

    R1 = pd.read_csv(path + "r1.csv", header=None, index_col=None).to_numpy()
    R2 = pd.read_csv(path + "r2.csv", header=None, index_col=None).to_numpy()
    R3 = pd.read_csv(path + "r3.csv", header=None, index_col=None).to_numpy()
    R4 = pd.read_csv(path + "r4.csv", header=None, index_col=None).to_numpy()
    receptor_feature = np.hstack((R1, R2, R3, R4))



    interaction = pd.read_csv(path+"interaction.csv", header=0, index_col=0).to_numpy()
    df = pd.read_csv(path + 'interaction.csv', header=0, index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])
    for i, name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]

    with open(path + 'receptor.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
    for i, name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    count0 = 0
    count1 = 0
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            temp = np.append(ligand_feature[i], receptor_feature[j])  # 加行和列的标签
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)
                LRI_name.append(df.index[i] + ' ' + df.columns[j])
                count1 += 1
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)
                LRI_name_z.append(df.index[i] + ' ' + df.columns[j])#LRI_name_z所有负样本的名字
                count0 += 1
    label00 = np.zeros((len(negative_feature), 1))#所有负样本添加标签

    LRI_known = LRI_name
    LRI_known = pd.DataFrame(LRI_known)
    LRI_known.to_csv(path + "LRI_name_known.csv", header=None, index=None)#保存已知正样本name

    # negative_feature = np.hstack([negative_feature, label00])
    # np.savetxt(path + "negative_feature.txt", negative_feature, delimiter=',')#保存所有负样本
    # LRI_name_z = pd.DataFrame(LRI_name_z)
    # LRI_name_z.to_csv(path + 'LRI_name_z.csv', header=None, index=None)#保存所有负样本name
    row = []
    col = []
    [row0, column0] = np.where(df.values == 0)

    rand = np.random.RandomState(seed)
    num = rand.randint(row0.shape, size=count1)
    for i in num:
        row.append(row0[i])
        col.append(column0[i])

    for i, j in zip(row, col):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = np.append(lig_tri_fea, rec_tri_fea)
        LRI_name.append(df.index[i] + ' ' + df.columns[j])
        negative_sample_feature.append(temp_f)

    #LRI_name = pd.DataFrame(LRI_name)
    #LRI_name.to_csv(path + 'LRI_name.csv', header=None, index=None)#同等数量0，1的数据集
    feature = np.vstack((positive_feature, negative_sample_feature))



    label1 = np.ones((len(positive_feature), 1))
    label0 = np.zeros((len(negative_sample_feature), 1))
    label = np.vstack((label1, label0))

    #XGBoost特征选择（降维）
    xgb_model = xgb.XGBClassifier()
    xgbresult1 = xgb_model.fit(feature, label.ravel())  # python 中的 ravel() 函数将数组多维度拉成一维数组
    feature_importance = xgbresult1.feature_importances_
    feature_number = -feature_importance
    H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    mask = H1[:500]
    train_data = feature[:, mask]

    #lightGBM特征选择（降维）
    # lgb_model = LGBMClassifier()
    # lgbresults = lgb_model.fit(feature, label.ravel())
    # feature_importance = lgbresults.feature_importances_
    # feature_number = -feature_importance
    # H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # mask = H1[:500]
    # train_data = feature[:, mask]


    # 标准化
    # std = StandardScaler()
    # train_data = std.fit_transform(train_data)
    # # #归一化
    # mm = MinMaxScaler()
    # train_data = mm.fit_transform(train_data)

    feature = np.hstack((train_data, label))
    # np.savetxt(path+"train.txt", feature, delimiter=',')
    feature = pd.DataFrame(feature)
    feature.to_csv(path + 'train.csv', header=None, index=None)
    #train表示训练用的数据集，其中包括正样本，负样本。test表示预测用的数据集，其中全部都是负样本，预测所有的0中哪些可能是1
    return feature




def Sample_generation800(seed,path):
    positive_feature = []
    negative_feature = []
    ligand_dict = {}
    receptor_dict = {}
    name_L = []
    name_R = []
    LRI_name = []
    feature = []
    LRI_name_z = []
    negative_sample_feature = []


    L1 = pd.read_csv(path+"l1.csv", header=None, index_col=None).to_numpy()
    L2 = pd.read_csv(path+"l2.csv", header=None, index_col=None).to_numpy()
    L3 = pd.read_csv(path+"l3.csv", header=None, index_col=None).to_numpy()
    L4 = pd.read_csv(path+"l4.csv", header=None, index_col=None).to_numpy()
    ligand_feature = np.hstack((L1, L2, L3, L4))

    R1 = pd.read_csv(path + "r1.csv", header=None, index_col=None).to_numpy()
    R2 = pd.read_csv(path + "r2.csv", header=None, index_col=None).to_numpy()
    R3 = pd.read_csv(path + "r3.csv", header=None, index_col=None).to_numpy()
    R4 = pd.read_csv(path + "r4.csv", header=None, index_col=None).to_numpy()
    receptor_feature = np.hstack((R1, R2, R3, R4))



    interaction = pd.read_csv(path+"interaction.csv", header=0, index_col=0).to_numpy()
    df = pd.read_csv(path + 'interaction.csv', header=0, index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])
    for i, name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]

    with open(path + 'receptor.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
    for i, name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    count0 = 0
    count1 = 0
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            temp = np.append(ligand_feature[i], receptor_feature[j])  # 加行和列的标签
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)
                LRI_name.append(df.index[i] + ' ' + df.columns[j])
                count1 += 1
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)
                LRI_name_z.append(df.index[i] + ' ' + df.columns[j])#LRI_name_z所有负样本的名字
                count0 += 1
    label00 = np.zeros((len(negative_feature), 1))#所有负样本添加标签

    LRI_known = LRI_name
    LRI_known = pd.DataFrame(LRI_known)
    LRI_known.to_csv(path + "LRI_name_known.csv", header=None, index=None)#保存已知正样本name

    # negative_feature = np.hstack([negative_feature, label00])
    # np.savetxt(path + "negative_feature.txt", negative_feature, delimiter=',')#保存所有负样本
    # LRI_name_z = pd.DataFrame(LRI_name_z)
    # LRI_name_z.to_csv(path + 'LRI_name_z.csv', header=None, index=None)#保存所有负样本name
    row = []
    col = []
    [row0, column0] = np.where(df.values == 0)

    rand = np.random.RandomState(seed)
    num = rand.randint(row0.shape, size=count1)
    for i in num:
        row.append(row0[i])
        col.append(column0[i])

    for i, j in zip(row, col):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = np.append(lig_tri_fea, rec_tri_fea)
        LRI_name.append(df.index[i] + ' ' + df.columns[j])
        negative_sample_feature.append(temp_f)

    #LRI_name = pd.DataFrame(LRI_name)
    #LRI_name.to_csv(path + 'LRI_name.csv', header=None, index=None)#同等数量0，1的数据集
    feature = np.vstack((positive_feature, negative_sample_feature))



    label1 = np.ones((len(positive_feature), 1))
    label0 = np.zeros((len(negative_sample_feature), 1))
    label = np.vstack((label1, label0))

    #XGBoost特征选择（降维）
    xgb_model = xgb.XGBClassifier()
    xgbresult1 = xgb_model.fit(feature, label.ravel())  # python 中的 ravel() 函数将数组多维度拉成一维数组
    feature_importance = xgbresult1.feature_importances_
    feature_number = -feature_importance
    H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    mask = H1[:800]
    train_data = feature[:, mask]

    #lightGBM特征选择（降维）
    # lgb_model = LGBMClassifier()
    # lgbresults = lgb_model.fit(feature, label.ravel())
    # feature_importance = lgbresults.feature_importances_
    # feature_number = -feature_importance
    # H1 = np.argsort(feature_number)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # mask = H1[:600]
    # train_data = feature[:, mask]


    # 标准化
    # std = StandardScaler()
    # train_data = std.fit_transform(train_data)
    # # #归一化
    # mm = MinMaxScaler()
    # train_data = mm.fit_transform(train_data)

    feature = np.hstack((train_data, label))
    # np.savetxt(path+"train.txt", feature, delimiter=',')
    feature = pd.DataFrame(feature)
    feature.to_csv(path + 'train.csv', header=None, index=None)
    #train表示训练用的数据集，其中包括正样本，负样本。test表示预测用的数据集，其中全部都是负样本，预测所有的0中哪些可能是1
    return feature

if __name__ == '__main__':
    Sample_generation(1, 'dataset/human/')
    # test_generation(1, 'data-seq/human/')









