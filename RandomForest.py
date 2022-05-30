import time
from DecisionTree import train_tree, prediction, entropy, gini
import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle

from sklearn import preprocessing
from collections import Counter
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from lib.auxiliary import printTree, oversample

class RandomForest:
    def __init__(self, data, impurity_func = entropy, max_depth = 20,min_samples_split = 2, trees = 25):
        self.data = data
        self.y = str(data.columns[-1])
        self.impurity_func = impurity_func
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = trees
        self.list_tree = []
    def create_fores(self):
        #Split data to any samples
        data = self.data.copy()
        list_samples = []
        size = (len(data) * 80) // 100
        for i in range(0, self.trees):
            list_samples.append(data.sample(size))
        
        #create trees
        t = 0
        for df in list_samples:
            tree = train_tree(df, self.y, self.impurity_func, self.max_depth, self.min_samples_split, 0)
            t += 1
            print('Tree {} was created'.format(t))
            self.list_tree.append(tree)
    
    def test_forest(self, df_test):
        list_pred = []
        total_0 = (df_test[str(df_test.columns[-1])] == 0).sum()
        total_1 = (df_test[str(df_test.columns[-1])] == 1).sum()
        print(total_0, total_1)
        for i in range(len(df_test)):
            list_result = []
            for tree in self.list_tree:
                list_result.append(int(prediction(tree, df_test.iloc[i])))

            counter = Counter(list_result)
            list_pred.append(counter.most_common(1)[0][0])
        return list_pred

    def take_prediction(self, df_row):
        '''
        df_row: a row of df (df.iloc[i])
        '''
        list_result = []
        for tree in self.list_tree:
            list_result.append(int(prediction(tree, df_row)))

        counter = Counter(list_result)
        return counter.most_common(1)[0][0]

if __name__ == '__main__':
    
    #Import data
    df = pd.read_table("./data/clean_data.csv", delimiter=',')
    df.drop('ID', axis = 1, inplace = True)
    df = oversample(df)
    df = shuffle(df)
    
    le = preprocessing.LabelEncoder()
    df["Income_type"] = le.fit_transform(df["Income_type"])
    df["Education_type"] = le.fit_transform(df["Education_type"])
    df["Family_status"] = le.fit_transform(df["Family_status"])
    df["Housing_type"] = le.fit_transform(df["Housing_type"])
    df["Occupation_type"] = le.fit_transform(df["Occupation_type"])
    df_train = df[:13481]
    df_test = df[13481:]

    #Train forest
    randomforest = RandomForest(df_train, entropy, 20, 2, 100)
    t1 = time.time()
    randomforest.create_fores()
    print("The random forest was created in {} seconds.".format(time.time() - t1))

    with open('./model/forest.model1', 'wb') as randomforestmodel:
        pickle.dump(randomforest, randomforestmodel)

    #Test fores
    with open('./model/forest.model1', 'rb') as randomforestmodel:
        my_forest = pickle.load(randomforestmodel)

    t2 = time.time()
    y_pred = my_forest.test_forest(df_test)
    print("The testing was done in {} seconds.".format(time.time() - t2))

    y_test = df_test['Y'].to_numpy()
    y_pred = np.array(y_pred).astype(int)
    print("Accuracy: {}%".format(accuracy_score(y_test, y_pred) * 100))
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()
    
    #Get prediction
    '''print("-----------Prediction-----------")
    print(df_test.iloc[18])
    print(randomforest.take_prediction(df_test.iloc[18]))'''

    