#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:44:11 2019

InClass Prediction Competition
TCD ML Comp. 2019/20 - Income Prediction (Ind.)
Goal: Predict the income.

@author: ECD4D148D85290A7B7D2

Kindly Refer to DataAnalysis.py or Jupyter Book - Cookbook.ipynb for explanation of each step.
Thank You!

Training Dataset: tcd ml 2019-20 income prediction training (with labels).csv
Training Dataset: tcd ml 2019-20 income prediction test (without labels).csv

Regression Model: Linear
Mean Absolute Error: 26337.441347977998
Mean Squared Error: 3213609856.9474673
Root Mean Squared Error: 54895.22631266166

IDE: Anaconda, Python 3.6
Platform: Ubuntu 16.04

"""

from DataAnalysis import dataEngineering
from DataPreprocess import DataPreprocess

trainDataPath = "/PATH/tcd ml 2019-20 income prediction training (with labels).csv"
testDataPath = "/PATH/tcd ml 2019-20 income prediction test (without labels).csv"
modelpath = "PATH_to_Pickled_Model"


def main():

    dp = DataPreprocess()

    traindata = dp.datasetLoad(trainDataPath)

    testdata = dp.datasetLoad(testDataPath)

    dataEngineering()

    traindata, target, testdata = dp.traindataPreprocessing(traindata, testdata)

    #DATA SPLIT FOR TRAINING AND TESTING
    X_train, X_test, y_train, y_test = dp.traindataPreparation(traindata, target)

    X_train, X_test, y_train, testdata = dp.columnEncoding(X_train, X_test, y_train, testdata)

    #TRAINING
    regressor = dp.modelTrain(X_train, y_train)
    #TESTING
    y_pred = dp.modelPredict(regressor, X_test)

    dp.modelAccuracy(y_test, y_pred)

    dp.modelSave(regressor)

    dp.modelLoad(modelpath)

    #PREDICTION ON TEST DATA
    y_pred_test = dp.modelPredict(regressor, testdata)

    dp.dataExport(testDataPath, y_pred_test)


if __name__ == '__main__':
  main()
