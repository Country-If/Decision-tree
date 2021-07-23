#!/usr/bin/env python
# -*- coding: UTF-8 -*-


__author__ = "Maylon"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import graphviz
from sklearn import datasets


def Bunch2dataframe(sklearn_dataset):
    """
    将sklearn数据集Bunch类型转成DataFrame
    :param sklearn_dataset: sklearn中的数据集
    :return: 处理后的dataframe，最后一列为标签列
    """
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)        # 追加一列标签列
    return df


def calcEnt(dataSet):
    """
    计算香农熵
    :param dataSet: 原始数据集(dataFrame)
    :return: 香农熵
    """
    tag_col = -1  # 标签所在列，根据实际dataFrame确定
    n = dataSet.shape[0]  # 数据集总行数
    iset = dataSet.iloc[:, tag_col].value_counts()  # 标签的所有类别
    p = iset / n  # 每一类标签所占比
    ent = (-p * np.log2(p)).sum()  # 计算信息熵
    return ent


def bestSplit(dataSet):
    """
    数据集最佳切分函数：根据信息增益选出最佳数据集切分的列
    :param dataSet: 原始数据集
    :return: 数据集最佳切分列的索引
    """
    baseEnt = calcEnt(dataSet)  # 计算原始熵
    bestGain = 0  # 初始化信息增益
    axis = -1  # 初始化最佳切分列，标签列，根据实际dataFrame确定
    for i in range(dataSet.shape[1] - 1):  # 对特征的每一列(除去标签列)进行循环
        levels = dataSet.iloc[:, i].value_counts().index  # 提取出当前列的所有值
        ents = 0  # 初始化子节点的信息熵
        for j in levels:  # 对当前列的每一个取值进行循环
            childSet = dataSet[dataSet.iloc[:, i] == j]  # 某一个子节点的dataframe
            ent = calcEnt(childSet)  # 计算某个子节点的信息熵
            ents += (childSet.shape[0] / dataSet.shape[0]) * ent  # 计算当前列的信息熵
        # print(f'第{i}列的信息熵为{ents}')
        infoGain = baseEnt - ents  # 计算当前列的信息增益
        # print(f'第{i}列的信息增益为{infoGain}')
        if infoGain > bestGain:  # 选择最大信息增益
            bestGain = infoGain
            axis = i
    return axis  # 返回最大信息增益所在列的索引


def dataSetSpilt(dataSet, axis, value):
    """
    按照给定的列划分数据集
    :param dataSet: 原始数据集
    :param axis: 指定的列索引
    :param value: 指定的属性值
    :return: 按照指定列索引和属性值切分后的数据集
    """
    col = dataSet.columns[axis]     # 指定列的索引
    SpiltDataSet = dataSet.loc[dataSet[col] == value, :].drop(col, axis=1)
    return SpiltDataSet


def createTree_ID3(dataSet):
    """
    ID3算法构建决策树
    :param dataSet:原始数据集，注意标签列不能是数值
    :return: 字典形式的树
    """
    tag_col = -1  # 标签所在列，根据实际dataFrame确定
    featlist = list(dataSet.columns)        # 提取出数据集所有的列
    classlist = dataSet.iloc[:, tag_col].value_counts()      # 获取类标签
    if classlist[0] == dataSet.shape[0] or dataSet.shape[1] == 1:       # 判断最多标签数目是否等于数据集行数或者数据集是否只有一列
        return classlist.index[0]       # 若是则返回类标签
    axis = bestSplit(dataSet)       # 确定当前最佳切分列的索引
    bestfeat = featlist[axis]       # 获取该索引对应的特征
    myTree = {bestfeat: {}}     # 采用字典嵌套的方式存储树信息
    del featlist[axis]      # 删除当前特征
    valuelist = set(dataSet.iloc[:, axis])      # 提取最佳切分列的所有属性值
    for value in valuelist:     # 对每一个属性值递归建树
        myTree[bestfeat][value] = createTree_ID3(dataSetSpilt(dataSet, axis, value))
    return myTree


def classify(inputTree, labels, testVec):
    """
    对一个测试实例进行分类
    :param inputTree: 已经生成的决策树
    :param labels: 存储选择的最优特征标签
    :param testVec: 测试数据列表，顺序对应原数据集
    :return: 分类结果
    """
    firstStr = next(iter(inputTree))        # 获取决策树第一个节点
    secondDict = inputTree[firstStr]        # 下一个字典
    featIndex = labels.index(firstStr)      # 第一个节点所在列的索引
    classLabel = secondDict[list(secondDict.keys())[0]]     # 标签初始化
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key], labels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def acc_classify(train, test, Tree):
    """
    对测试集进行预测，并返回预测后的结果
    :param train: 训练集
    :param test: 测试集
    :param Tree: 决策树
    :return: 预测好分类的测试集和准确率(tuple)
    """
    labels = list(train.columns)        # 数据集所有的名称
    row_index = test.index.to_list()
    result = pd.DataFrame(None, index=row_index, columns=["predict"])       # 初始化result，dataframe类型
    for i in range(test.shape[0]):      # 对测试集中每一行数据(每一个实例)进行循环
        testVec = test.iloc[i, :-1]     # 取出每行的数据部分；标签列是最后一列，根据实际dataframe确定
        classLabel = classify(Tree, labels, testVec)       # 预测该实例的分类
        result.iloc[i, 0] = classLabel      # 将分类结果追加到result列表中
    test = pd.concat([test, result], axis=1)        # 拼接两个dataframe
    acc = (test.iloc[:, -1] == test.iloc[:, -2]).mean()     # 计算准确率；最后一列为预测结果，倒数第二列为标签列
    return test, acc     # 返回测试集和准确率


def save_tree(Tree, filename="mytree.npy"):
    """
    保存决策树
    :param filename: 保存为*.npy文件
    :param Tree: 所构建的决策树
    """
    try:
        np.save(filename, Tree)
        print("Tree Saved in " + filename)
    except Exception as e:
        print(e)
        print("Failed to Save the Tree.")


def load_tree(filename="mytree.npy"):
    """
    加载决策树
    :param filename: 读取的*.npy文件
    :return: 决策树
    """
    try:
        Tree = np.load(filename, allow_pickle=True).item()
        return Tree
    except Exception as e:
        print(e)
        print("Failed to Load the Tree.")


# def split_dataset(dataSet_data, dataSet_target, test_size):
#     """
#
#     :param dataSet_data: 数据集的数据
#     :param dataSet_target: 数据集的标签
#     :param test_size: 切分数据集的比例
#     :return: 切分后的训练集和测试集(数据和标签分开)<class 'numpy.ndarray'>
#     """
#     train_data, test_data, train_target, test_target = train_test_split(dataSet_data,
#                                                                         dataSet_target,
#                                                                         test_size=test_size)
#     return train_data, test_data, train_target, test_target


def get_test_dataset():
    """
    调试用，固定训练集和测试集
    :return: None
    """
    data = datasets.load_iris()
    dataset = Bunch2dataframe(data)
    target_col = -1
    for i in range(len(dataset)):
        if dataset.iloc[i, target_col] == 0:
            dataset.iloc[i, target_col] = 'a'
        elif dataset.iloc[i, target_col] == 1:
            dataset.iloc[i, target_col] = 'b'
        elif dataset.iloc[i, target_col] == 2:
            dataset.iloc[i, target_col] = 'c'
    print(dataset)
    train, test = train_test_split(dataset, test_size=0.3)
    train.to_csv("train.csv", index=None)
    test.to_csv("test.csv", index=None)


def ID3():
    # get_test_dataset()        # 调试
    # train = pd.read_csv("train.csv")      # 调试，固定训练集
    # test = pd.read_csv("test.csv")        # 调试，固定测试集
    data = datasets.load_iris()     # 加载数据集
    dataset = Bunch2dataframe(data)
    target_col = -1
    # 标签列不可为数值，故对标签列进行处理
    for i in range(len(dataset)):
        if dataset.iloc[i, target_col] == 0:
            dataset.iloc[i, target_col] = 'a'
        elif dataset.iloc[i, target_col] == 1:
            dataset.iloc[i, target_col] = 'b'
        elif dataset.iloc[i, target_col] == 2:
            dataset.iloc[i, target_col] = 'c'
    print(dataset)
    train, test = train_test_split(dataset, test_size=0.3)      # 切分训练集和测试集
    mytree = createTree_ID3(train)      # 构建决策树
    save_tree(mytree)
    tree_model = load_tree()
    print(tree_model)
    test_result, score = acc_classify(train, test, tree_model)      # 对测试集进行预测并给出准确率
    print(test_result)
    print(score)


def Draw_tree(clf, filename, feature_names=None, class_names=None):
    """
    绘制决策树并保存为*.pdf文件
    :param clf: 训练后的模型
    :param filename: 保存的文件名
    :param feature_names: 特征名
    :param class_names: 标签名
    :return: None
    """
    dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename)
    print("Done.")


def best_depth_tree(train, test):
    """
    调参得到最佳的max_depth值并返回对应训练后的模型
    :param train: 训练集
    :param test: 测试集
    :return: 训练后的模型列表和测试集预测准确率最大值的索引
    """
    train_score_list = []
    test_score_list = []
    clf_list = []
    max_test_depth = 10     # 最大树深(超参数上限)
    train_data = train.iloc[:, :-1]
    train_target = train.iloc[:, -1]
    test_data = test.iloc[:, :-1]
    test_target = test.iloc[:, -1]
    for i in range(max_test_depth):
        clf = DecisionTreeClassifier(criterion="entropy",
                                     max_depth=i+1,
                                     random_state=30,
                                     splitter="random"
                                     )
        clf = clf.fit(train_data, train_target)     # 训练模型
        score_train = clf.score(train_data, train_target)       # 训练集预测准确率
        score = clf.score(test_data, test_target)       # 测试集预测准确率
        train_score_list.append(score_train)
        test_score_list.append(score)
        clf_list.append(clf)
    plt.plot(range(1, max_test_depth+1), train_score_list, color="blue", label="train")        # 绘制分数曲线
    plt.plot(range(1, max_test_depth+1), test_score_list, color="red", label="test")
    plt.legend()
    plt.show()
    return clf_list, test_score_list.index(max(test_score_list))


def sklearn():
    data = datasets.load_wine()     # 加载数据集
    dataset = Bunch2dataframe(data)     # 转换成dataframe类型进行处理，最后一列为标签列
    train, test = train_test_split(dataset)     # 切分训练集和测试集
    feature_names = dataset.columns[:-1]        # 获取特征名
    clf_list, i = best_depth_tree(train, test)      # 训练模型
    print("max_depth: " + str(i+1))
    clf = clf_list[i]     # 选取测试集预测准确率最大值的模型
    Draw_tree(clf, "wine", feature_names=feature_names)     # 绘制决策树


if __name__ == '__main__':
    # ID3()
    sklearn()
