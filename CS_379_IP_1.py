#Adam Mick CS 379 Indiviual project 1

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np


def main():

    ######################
    #Cleaning the dataset#
    ######################

    #loading the data from the spreadsheet
    numericalData = pd.read_excel("dataset.xls", usecols=['survived', 'age', 'fare'])
    categoryData = pd.read_excel("dataset.xls", usecols=['sex','embarked', 'cabin'])

    categoryData.cabin.fillna(categoryData.cabin.value_counts().idxmax(), inplace = True) #replace the null values in cabin with the max count
    categoryData.embarked.fillna(categoryData.embarked.value_counts().idxmax(), inplace = True)#replace the null values in the embarked data

    ######check for null values
    #print(numericalData.isnull().sum())

    #transform the catagorey data into numbers using the sklearn library
    from sklearn.preprocessing import LabelEncoder
    label = LabelEncoder()
    categoryData = categoryData.apply(label.fit_transform)

    #cleaning numerical data
    #print(numericalData.isna().sum()) checking for null values in the data

    numericalData.age.fillna(numericalData.age.mean(), inplace = True) #filling the null age values with the mean
    numericalData.fare.fillna(numericalData.fare.mean(), inplace = True)

        #merge the two dataframes into one
    finalData = pd.concat([categoryData, numericalData], axis = 1)
    finalData.head()

    KNN(numericalData, categoryData, finalData)
    Kmeans(finalData)

    return

def KNN(numericalData, categoryData, finalData):

    ###########################################
    #Implementing the KNN supervised algorithm#
    ###########################################

    X = finalData.drop(['survived'], axis = 1) #defining our X variable
    Y = finalData['survived'] #our Y variable is the target for our supervised algorithm

    X_train = np.array(X[0:int(0.80*len(X))]) #80% of the data is used for training
    Y_train = np.array(Y[0:int(0.80*len(Y))])
    X_test = np.array(X[int(0.80*len(X)):])#20% of the data will be used as test data
    Y_test = np.array(Y[int(0.80*len(Y)):])
    len(X_train), len(Y_train), len(X_test), len(Y_test)

    from sklearn.neighbors import KNeighborsClassifier

    KNN = KNeighborsClassifier(n_neighbors = 5)#Running our data through the KNN algorthm
    KNN_fit = KNN.fit(X_train, Y_train)
    KNN_pred = KNN_fit.predict(X_test)
    print(KNN_pred)

    ##########################
    #Evaluating our algorithm#
    ##########################

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(Y_test, KNN_pred))
    print(classification_report(Y_test, KNN_pred)) #using a k value of 5 results in an 84% accuracy rate on the dataset

    return


def Kmeans(finalData):

    ###########################################################
    #Implementing the Kmeans unsupervised clustering algorithm#
    ###########################################################

    from kneed import KneeLocator 
    from sklearn.cluster import KMeans #importing some required libraries
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import DBSCAN

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(finalData)

    kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
    )

    kmeans.fit(scaled_features)

    # Instantiate k-means and dbscan algorithms
    kmeans = KMeans(n_clusters=2)
    dbscan = DBSCAN(eps=0.3)

    # Fit the algorithms to the features
    kmeans.fit(scaled_features)
    dbscan.fit(scaled_features)

    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
    }

     #A list holds the SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
        
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    return

main()
