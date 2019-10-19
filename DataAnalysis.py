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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


trainDataPath = "/PATH/tcd ml 2019-20 income prediction training (with labels).csv"
traindata = pd.read_csv(trainDataPath)



def dataEngineering():
    
    
    traindata = pd.read_csv(trainDataPath)
    missing_data = traindata.isnull().sum()
    total_missing_data = missing_data.sum() #23301
    total_cells = np.product(traindata.shape)  #1231923
    perc_missing_data = (total_missing_data/total_cells)*100
    print("Out of ", total_cells," total cells, " , total_missing_data," cells have missing data")
    print("Missing data petrcentage: ", round(perc_missing_data,2), " %")
    print("Missing data stats: ", missing_data)

    """
    Out of  1231923  total cells,  23301  cells have missing data

    Missing data petrcentage:  1.73  %

    Missing data stats:
    Year of Record        441
    Gender               7432
    Age                   494
    Country                 0
    Size of City            0
    Profession            322
    University Degree    7370
    Wears Glasses           0
    Hair Color           7242
    Body Height [cm]        0
    Income in EUR           0
    dtype: int64
    """

    # TO VISUALISE THE PERCENTAGE OF MISSING VALUES FOR EACH COLUMN
    print("Missing percentage per columns:", np.around(traindata.isnull().mean()*100, decimals = 2))

    """
    Missing percentage per columns:

    Year of Record       0.39
    Gender               6.64 **
    Age                  0.44
    Country              0.00
    Size of City         0.00
    Profession           0.29
    University Degree    6.58 **
    Wears Glasses        0.00
    Hair Color           6.47 **
    Body Height [cm]     0.00
    Income in EUR        0.00

    """

    # CARDINALITY

    #COUNT OF UNIQUE VALUES PER COLUMN
    category_cols = ['Gender', 'Country', 'Profession', 'University Degree', 'Hair Color']
    for col in category_cols:
        print('variable: ', col, ' number of labels: ', traindata[col].nunique())

    print('total records: ', len(traindata))

    '''
    variable:  Gender  number of labels:  5
    variable:  Country  number of labels:  160
    variable:  Profession  number of labels:  1340
    variable:  University Degree  number of labels:  5
    variable:  Hair Color  number of labels:  6
    total records:  111993
    '''

    #HOW FREQUENTLY EACH LABEL APPEARS IN THE DATASET

    total_records = len(traindata)
    for col in category_cols:
        print("For Column:", traindata[col].value_counts())
        temp_df = pd.Series(traindata[col].value_counts() / total_records)
        fig = temp_df.sort_values(ascending=False).plot.bar()
        fig.set_xlabel(col)
        fig.set_ylabel('Percentage of records')
        #plt.show()
        plt.savefig('Percentage of Records per category')


    plt.scatter(traindata['Age'], traindata["Income in EUR"], color= 'red')
    plt.title('Age Vs. Income in EUR')
    plt.ylabel('Income in EUR', fontsize=12)
    plt.xlabel('Age', fontsize=12)
    plt.savefig('Age Vs. Income in EUR.png')

    plt.scatter(traindata['Year of Record'], traindata["Income in EUR"], color= 'red')
    plt.title('Year of Record Vs. Income in EUR')
    plt.ylabel('Income in EUR', fontsize=12)
    plt.xlabel('Year of Record', fontsize=12)
    plt.savefig('Year of Record Vs. Income in EUR.png')

    plt.scatter(traindata['Wears Glasses'], traindata["Income in EUR"], color= 'red')
    plt.title('Wears Glasses Vs. Income in EUR')
    plt.ylabel('Income in EUR', fontsize=12)
    plt.xlabel('Wears Glasses', fontsize=12)
    plt.savefig('Wears Glasses Vs. Income in EUR.png')

    plt.scatter(traindata['Body Height [cm]'], traindata["Income in EUR"], color= 'red')
    plt.title('Body Height Vs. Income in EUR')
    plt.ylabel('Income in EUR', fontsize=12)
    plt.xlabel('Body Height [cm]', fontsize=12)
    plt.savefig('Body Height Vs. Income in EUR.png')

    plt.scatter(traindata['Size of City'], traindata["Income in EUR"], color= 'red')
    plt.title('Size of City Vs. Income in EUR')
    plt.ylabel('Income in EUR', fontsize=12)
    plt.xlabel('Size of City', fontsize=12)
    plt.savefig('Size of City Vs. Income in EUR.png')


    ####################
    #         AGE      # RIGHT SKEWED-DISTRIBUTION. THUS, MEAN DOESNT REPRESENT THE CENTER OF NORMAL DISTRIBUTION
    ####################

    fig = plt.figure()
    ax = fig.add_subplot(111)
    traindata.Age.plot(kind='kde', ax=ax)
    plt.savefig("Age Distribution.png")

    traindata.Age.hist(bins=50, figsize=(10,10))
    plt.savefig("Age Histogram")

    print('Original Age variable variance: ', traindata.Age.var())
    #Original variable variance:  257.1755589176853
    #traindata[['Age', 'Income in EUR']].cov()

    #SINCE RIGHT SKEWED DISTRIBUTION, PERFORM MEDIAN IMPUTATION AS MEDIAN BEST REPRESENTS THE DISTRIBUTION AND NOT MEAN

    traindata['AGE_MEDIAN_IMPUTED'] = traindata.Age.fillna(traindata.Age.median())
    #PLOT KDE
    fig = plt.figure()
    ax = fig.add_subplot(111)
    traindata.Age.plot(kind='kde', ax=ax)
    traindata['AGE_MEDIAN_IMPUTED'].plot(kind='kde', ax=ax, color='red')
    plt.savefig("Age Median Imputed Distribution.png")


    print('Median Imputed Age variable variance: ', traindata['AGE_MEDIAN_IMPUTED'].var())
    #Original variable variance:  257.1755589176853
    traindata[['AGE_MEDIAN_IMPUTED', 'Income in EUR']].cov()


    # VARIABLE TRANSFORMATION

    #histogram and Q-Q plot
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    traindata['Age'].hist(bins=30)
    plt.subplot(1, 2, 2)
    stats.probplot(traindata['Age'], dist="norm", plot=plt)
    plt.savefig("Age Variable Distribution before sqaure root transformation.png")

    temp = pd.DataFrame()
    temp['temp'] = traindata['AGE_MEDIAN_IMPUTED']**(1/2)

    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    temp['temp'].hist(bins=30)
    plt.subplot(1, 2, 2)
    stats.probplot(temp['temp'], dist="norm", plot=plt)
    plt.savefig("Age Variable Distribution after sqaure root transformation.png")
    temp = temp.drop("temp", axis=1)
    traindata.drop('AGE_MEDIAN_IMPUTED', 1)


    ####################
    # Year of Record   #  RIGHT SKEWED-DISTRIBUTION. THUS, MEAN DOESNT REPRESENT THE CENTER OF NORMAL DISTRIBUTION
    ####################

    fig = plt.figure()
    ax = fig.add_subplot(111)
    traindata['Year of Record'].plot(kind='kde', ax=ax)
    plt.savefig("Year of Record Distribution.png")

    traindata['Year of Record'].hist(bins=50, figsize=(10,10))
    plt.savefig("Year of Record Histogram.png")

    print('Original Year of Record variable variance: ', traindata['Year of Record'].var())
    #Original variable variance:  257.1755589176853
    #traindata[['Year of Record', 'Income in EUR']].cov()

    #SINCE RIGHT SKEWED DISTRIBUTION, PERFORM MEDIAN IMPUTATION AS MEDIAN BEST REPRESENTS THE DISTRIBUTION AND NOT MEAN

    traindata['YEAR_MEDIAN_IMPUTED'] = traindata['Year of Record'].fillna(traindata['Year of Record'].median())
    #PLOT KDE
    fig = plt.figure()
    ax = fig.add_subplot(111)
    traindata['Year of Record'].plot(kind='kde', ax=ax)
    traindata['YEAR_MEDIAN_IMPUTED'].plot(kind='kde', ax=ax, color='red')
    plt.savefig("Year of Record Median Imputed Distribution.png")


    print('Median Imputed Year of Record variable variance: ', traindata['YEAR_MEDIAN_IMPUTED'].var())
    #Original variable variance:  257.1755589176853
    traindata[['YEAR_MEDIAN_IMPUTED', 'Year of Record']].cov()


    # VARIABLE TRANSFORMATION

    #histogram and Q-Q plot
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    traindata['Year of Record'].hist(bins=30)
    plt.subplot(1, 2, 2)
    stats.probplot(traindata['Year of Record'], dist="norm", plot=plt)
    plt.savefig("Year of Record Variable Distribution before sqaure root transformation.png")

    temp = pd.DataFrame()
    temp['temp'] = traindata['YEAR_MEDIAN_IMPUTED']**(1/2)

    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    temp['temp'].hist(bins=30)
    plt.subplot(1, 2, 2)
    stats.probplot(temp['temp'], dist="norm", plot=plt)
    plt.savefig("Year of Record Variable Distribution after sqaure root transformation.png")

    temp = temp.drop("temp", axis=1)
    traindata.drop('YEAR_MEDIAN_IMPUTED', 1)
