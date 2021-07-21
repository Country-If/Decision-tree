#!/usr/bin/env python
# -*- coding: UTF-8 -*-


__author__ = "Maylon"

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import graphviz
from sklearn import datasets


def Bunch2dataframe(sklearn_dataset):
    """
    将sklearn数据集Bunch类型转成DataFrame
    :param sklearn_dataset: sklearn中的数据集
    :return: 处理后的dataframe
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


def createDataSet():
    """
    :return: DataFrame
    """
    row_data = {'no surfacing': [1, 1, 1, 0, 0],
                'flippers': [1, 1, 0, 1, 1],
                'fish': ['yes', 'yes', 'no', 'no', 'no']}
    dataSet = pd.DataFrame(row_data)
    return dataSet


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
    :param dataSet:原始数据集
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
    classLabel = ""
    firstStr = next(iter(inputTree))        # 获取决策树第一个节点
    secondDict = inputTree[firstStr]        # 下一个字典
    featIndex = labels.index(firstStr)      # 第一个节点所在列的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key], labels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def acc_classify(train, test):
    """
    对测试集进行预测，并返回预测后的结果
    :param train: 训练集
    :param test: 测试集
    :return: 预测好分类的测试集和准确率(tuple)
    """
    inputTree = createTree_ID3(train)       # 根据训练集生成决策树
    labels = list(train.columns)        # 数据集所有的名称
    result = []
    for i in range(test.shape[0]):      # 对测试集中每一行数据(每一个实例)进行循环
        testVec = test.iloc[i, :-1]     # 取出每行的数据部分；标签列是最后一列，根据实际dataframe确定
        classLabel = classify(inputTree, labels, testVec)       # 预测该实例的分类
        result.append(classLabel)       # 将分类结果追加到result列表中
    test['predict'] = result        # 将预测结果追加到测试集最后一列
    acc = (test.iloc[:, -1] == test.iloc[:, -2]).mean()     # 计算准确率；最后一列为预测结果，倒数第二列为标签列
    return test, acc     # 返回测试集和准确率


def save_tree(filename, tree):
    """
    保存决策树
    :param filename: 保存为*.npy文件
    :param tree: 所构建的决策树
    """
    np.save(filename, tree)


def load_tree(filename):
    """
    加载决策树
    :param filename: 读取的*.npy文件
    :return: 决策树
    """
    tree = np.load(filename, allow_pickle=True).item()
    return tree


def split_dataset(dataSet_data, dataSet_target, test_size):
    """

    :param dataSet_data: 数据集的数据
    :param dataSet_target: 数据集的标签
    :param test_size: 切分数据集的比例
    :return: 切分后的训练集和测试集(数据和标签分开)<class 'numpy.ndarray'>
    """
    train_data, test_data, train_target, test_target = train_test_split(dataSet_data, dataSet_target, test_size=test_size)
    return train_data, test_data, train_target, test_target


def Make_tree(train_data, test_data, train_target, test_target, criterion='gini'):
    clf = DecisionTreeClassifier(criterion=criterion)
    clf = clf.fit(train_data, train_target)
    score = clf.score(test_data, test_target)
    return clf, score


def Draw_tree(clf, filename, feature_names=None, class_names=None):
    dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename)
    print("Done.")


if __name__ == '__main__':
    dataset = createDataSet()
    print(dataset)
    # Ent = calcEnt(dataset)
    # print(f'香农熵的值为{Ent}')
    # column = bestSplit(dataset)
    # print(f'最大信息增益所在列的索引为{column}')
    # mytree = createTree_ID3(dataset)
    # print(mytree)
    # np.save('myTree.npy', mytree)
    # save_tree('myTree.npy', mytree)
    # read_myTree = np.load('myTree.npy', allow_pickle=True).item()
    read_myTree = load_tree('myTree.npy')
    print(read_myTree)
