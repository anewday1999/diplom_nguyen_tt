import numpy as np
import random
import pandas as pd
import pickle
from collections import Counter
import random, time
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn import preprocessing
from sklearn.utils import shuffle

def downsample(df:pd.DataFrame, label_col_name:str) -> pd.DataFrame:
    # find the number of observations in the smallest group
    nmin = df[label_col_name].value_counts().min()
    return (df
            # split the dataframe per group
            .groupby(label_col_name)
            # sample nmin observations from each group
            .apply(lambda x: x.sample(nmin))
            # recombine the dataframes
            .reset_index(drop=True)
            )



def oversample(df):
    print("=======")
    classes = df.Y.value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df['Y'] == key]) 
        print(len(df[df['Y'] == key]))
    print(len(classes_list))
    classes_sample = []
    for i in range(1,len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    print()
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe,classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df

df = pd.read_table("./data/clean_data.csv", delimiter=',')
print(df['Y'].value_counts())

'''columns = df.columns
for cl in columns:
    str1 = cl
    print(np.unique(df[str1].values))
    print(len(df[df['Y'] == 0][df[str1] == 0]))
    print(len(df[df['Y'] == 0][df[str1] == 1]))
    print(len(df[df['Y'] == 1][df[str1] == 0]))
    print(len(df[df['Y'] == 1][df[str1] == 1]))
    g = sns.histplot(data=df, x=str1, hue="Y", multiple="dodge",bins=15)

    plt.show()
    a = input()'''

df = oversample(df)
str1 = 'Y'
'''print(np.unique(df[str1].values))
print(len(df[df['Y'] == 0][df[str1] == 0]))
print(len(df[df['Y'] == 0][df[str1] == 1]))
print(len(df[df['Y'] == 1][df[str1] == 0]))
print(len(df[df['Y'] == 1][df[str1] == 1]))
g = sns.histplot(data=df, x=str1, hue="Y", multiple="dodge",bins=15)

plt.show()
df = shuffle(df)
df.drop('ID', axis = 1, inplace = True)

le = preprocessing.LabelEncoder()
df["Income_type"] = le.fit_transform(df["Income_type"])
df["Education_type"] = le.fit_transform(df["Education_type"])
df["Family_status"] = le.fit_transform(df["Family_status"])
df["Housing_type"] = le.fit_transform(df["Housing_type"])
df["Occupation_type"] = le.fit_transform(df["Occupation_type"])'''

