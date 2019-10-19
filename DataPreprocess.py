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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_engine import missing_data_imputers as mdi
from sklearn.model_selection import train_test_split
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle


class DataPreprocess(object):


    def traindataPreprocessing(self, traindata, testdata):

        traindata = traindata.drop('Instance', 1)
        testdata = testdata.drop('Instance', 1)

        traindata = traindata[traindata['Income in EUR'] >= 0]
        traindata = traindata[traindata['Income in EUR'] <= 2500000]
        target = traindata['Income in EUR']

        traindata = traindata.drop('Size of City', 1)
        testdata = testdata.drop('Size of City', 1)

        traindata = traindata.drop('Income in EUR', 1)
        testdata = testdata.drop('Income', 1)

        traindata, testdata = self.imputeAge(traindata, testdata)
        traindata, testdata = self.imputeYearofRecord(traindata, testdata)
        traindata, testdata = self.transformBodyHeight(traindata, testdata)
        traindata, testdata = self.imputeGender(traindata, testdata)
        traindata, testdata = self.imputeUniversity(traindata, testdata)
        traindata, testdata = self.imputeProfession(traindata, testdata)
        traindata, testdata = self.imputeHairColor(traindata, testdata)

        return traindata, target, testdata

    def imputeAge(self, traindata, testdata):


        imputer = mdi.MeanMedianImputer(imputation_method='median',
                                variables=['Age'])

        imputer.fit(traindata)
        print("Replacing NaNs with Median in Column Age: ",imputer.imputer_dict_)
        traindata = imputer.transform(traindata)
        testdata = imputer.transform(testdata)

        traindata['Age'] = traindata['Age']**(1/2)
        testdata['Age'] = testdata['Age']**(1/2)

        return traindata, testdata


    def imputeYearofRecord(self, traindata, testdata):


        imputer = mdi.MeanMedianImputer(imputation_method='median',
                                variables=['Year of Record'])

        imputer.fit(traindata)
        print("Replacing NaNs with Median in Column Year of Record: ",imputer.imputer_dict_)
        traindata = imputer.transform(traindata)
        testdata = imputer.transform(testdata)

        traindata['Year of Record'] = traindata['Year of Record']**(1/2)
        testdata['Year of Record'] = testdata['Year of Record']**(1/2)

        return traindata, testdata

    def transformBodyHeight(self, traindata, testdata):

        traindata['Body Height [cm]'] = traindata['Body Height [cm]'] / 100
        testdata['Body Height [cm]'] = testdata['Body Height [cm]'] / 100

        return traindata, testdata

    def imputeGender(self, traindata, testdata):

        traindata.Gender = traindata.Gender.replace( np.NaN ,'missing_gender')
        traindata.Gender = traindata.Gender.replace( 'unknown' ,'missing_gender')
        traindata.Gender = traindata.Gender.replace( '0' ,'missing_gender')

        testdata.Gender = testdata.Gender.replace( np.NaN ,'missing_gender')
        testdata.Gender = testdata.Gender.replace( 'unknown' ,'missing_gender')
        testdata.Gender = testdata.Gender.replace( '0' ,'missing_gender')

        return traindata, testdata

    def imputeProfession(self, traindata, testdata):

        traindata['Profession'] = traindata['Profession'].replace( np.NaN ,'missing_prof')

        testdata['Profession'] = testdata['Profession'].replace( np.NaN ,'missing_prof')

        return traindata, testdata

    def imputeUniversity(self, traindata, testdata):

        traindata['University Degree'] = traindata['University Degree'].replace( np.NaN ,'missing_degree')
        traindata['University Degree'] = traindata['University Degree'].replace( '0' ,'missing_degree')

        testdata['University Degree'] = testdata['University Degree'].replace( np.NaN ,'missing_degree')
        testdata['University Degree'] = testdata['University Degree'].replace( '0' ,'missing_degree')


        return traindata, testdata

    def imputeHairColor(self, traindata, testdata):

        traindata['Hair Color'] = traindata['Hair Color'].replace( np.NaN ,'missing_hair')
        traindata['Hair Color'] = traindata['Hair Color'].replace( '0' ,'missing_hair')
        traindata['Hair Color'] = traindata['Hair Color'].replace( 'Unknown' ,'missing_hair')

        testdata['Hair Color'] = testdata['Hair Color'].replace( np.NaN ,'missing_hair')
        testdata['Hair Color'] = testdata['Hair Color'].replace( '0' ,'missing_hair')
        testdata['Hair Color'] = testdata['Hair Color'].replace( 'Unknown' ,'missing_hair')


        return traindata, testdata

    def traindataPreparation(self, traindata, target):

        X_train, X_test, y_train, y_test = train_test_split(traindata,
                                                            target,
                                                            test_size=0.2,
                                                            random_state=0)


        return X_train, X_test, y_train, y_test

    def columnEncoding(self, X_train, X_test, y_train, testdata):

        ohe = OneHotCategoricalEncoder(top_categories=None,
                                       variables=['Gender', 'University Degree','Hair Color','Profession','Country'],
                                       drop_last=True)
        ohe.fit(X_train, y_train)
        X_train = ohe.transform(X_train)
        X_test = ohe.transform(X_test)
        testdata = ohe.transform(testdata)

        return X_train, X_test, y_train, testdata

    def modelTrain(self, X_train, y_train):

        regressor = LinearRegression()
        regressor.fit(X_train, np.log(y_train))

        return regressor

    def modelPredict(self, regressor, X_test):

        y_pred = regressor.predict(X_test)
        y_pred = np.exp(y_pred)

        return y_pred

    def modelAccuracy(self, y_test, y_pred):

        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        df1 = df.head(25)
        df1.plot(kind='bar',figsize=(10,8))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.savefig('Actual Vs. Prediction.png')
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


    def datasetLoad(self, path):

        dataset = pd.read_csv(path)
        return dataset

    def dataExport(self, path, y_pred):

        testdata = self.datasetLoad(path)
        outframe = pd.DataFrame({'Instance': testdata.Instance, 'Income': y_pred })
        outframe = outframe.round(2)
        outframe.to_csv ('ECD4D148D85290A7B7D2_Predictions_LastUpload.csv', index = False, header=True)

    def modelSave(self, regressor):

        filename = 'TCD_KaggleComp_Regressor_Model.sav'
        pickle.dump(regressor, open(filename, 'wb'))

    def modelLoad(filename):

        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model
