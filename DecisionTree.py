import pandas as pd
import numpy as np
import time
import pickle
from sklearn import preprocessing
from node import Node
from collections import Counter
from sklearn.utils import shuffle
from lib.auxiliary import printTree, oversample
'''
gini va entropy dung de kiem tra do tinh khiet cua 1 var
ig de kiem tra xem dung bien nao de chia la tot nhat
vay lam sao biet gia tri nao de chia cua bien do la tot nhat, xem ham maxig
dung bien nao de chia.
'''

def gini(y):
  '''
  Return Gini Impurity. 
  y: variable.
  '''
  try:
    p = y.value_counts() / y.shape[0]
    gini = 1 - np.sum(p ** 2)
    return(gini)
  except Exception as e:
    print(e)
def entropy(var):
  '''
  Input: Pandas Series.
  Return: entropy. 
  Y: variable.
  '''
  try:
    a = var.value_counts() / var.shape[0]
    entropy = np.sum(-a * np.log2(a))
    return(entropy)
  except Exception as e:
    print(e)

def information_gain(target_var, mask, func):
  '''
  Return Information Gain of a variable given a loss function.
  target_var: target variable.
  mask: index and boolean of one part of data splited.
  func: function to be used to calculate information gain.
  '''
  a = sum(mask)
  b = mask.shape[0] - a
  if (a == 0 or b ==0): 
    ig = 0
  else:
    #for classification
    ig = func(target_var) - a / (a + b) * func(target_var[mask]) - b / (a + b) * func(target_var[-mask])
  return ig

def max_information_gain_split(var_testing, target_var, func):
  '''
  Return: best ig, and best value to split in this variable
  var_testing: predictor variable.
  target_var: target variable.
  func: function to be used to calculate the best split.
  '''
  split_value = []
  ig = [] 
  # Create options
  options = var_testing.sort_values().unique()
  #print("Length options: {}".format(len(options)))
  # Calculate ig for all values
  for val in options:
    mask =   var_testing <= val

    #val_ig = information_gain(target_var, mask, func)
    val_ig = information_gain(target_var=target_var, mask = mask, func=func)
    # Append results
    ig.append(val_ig)
    split_value.append(val)

  if len(ig) == 0:
    return None, None
  else:
    best_ig = max(ig)
    best_ig_index = ig.index(best_ig)
    best_split = split_value[best_ig_index]
    return best_ig, best_split

def get_best_option(df, var_target_name, func):
  '''
  Return best variable, it's ig and best value to split, and type of value in this variable
  '''
  variable = ''
  max_best_ig = 0
  split_value = 0
  for var in df.columns:
    if str(var) != var_target_name:
      best_ig, best_split = max_information_gain_split(df[str(var)], df[var_target_name], func)
      if (best_ig == None):
        return None, None, None
      if best_ig > max_best_ig:
        max_best_ig = best_ig
        variable = str(var)
        split_value = best_split
  if max_best_ig != 0:
    return variable, max_best_ig, split_value
  else:
    return None, None, None

def make_split(variable, value, data):

  data_1 = data[data[variable] <= value]
  data_2 = data[(data[variable] <= value) == False]
  
  return data_1,data_2

def train_tree(data, y, impurity_func = entropy, max_depth = 5,min_samples_split = 10, counter=0):
  #print("depth", counter)
  if len(data) >= min_samples_split and counter <= max_depth:
    variable, max_best_ig, split_value = get_best_option(data, y, func = impurity_func)
    if max_best_ig is not None:
      left, right = make_split(variable, split_value, data)
      if left.empty or right.empty:
        return Node(
          value=Counter(data[y]).most_common(1)[0][0]
        )
      yes_answer = train_tree(left, y, impurity_func, max_depth, min_samples_split, counter + 1)
      no_answer = train_tree(right, y, impurity_func, max_depth, min_samples_split, counter + 1)
      return Node(
          feature = variable, 
          threshold = split_value, 
          data_left = yes_answer, 
          data_right = no_answer, 
          gain = max_best_ig
      )
  return Node(
          value=Counter(data[y]).most_common(1)[0][0]
      )

def prediction(tree, row):
  if tree.value != None:
    return tree.value
  if row[tree.feature] <= tree.threshold:
    return prediction(tree.data_left, row)
  
  if row[tree.feature] > tree.threshold:
    return prediction(tree.data_right, row)

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

  df_train = df[:8000]
  df_test = df[8000:11000]
  '''t1 = time.time()
  decisiones = train_tree(df_train, str(df_train.columns[-1]), entropy, 30, 2, 0)
  print("Training time: ", time.time() - t1)
  #printTree(decisiones)
  with open('./model/decision.tree', 'wb') as decisiontree:
    pickle.dump(decisiones, decisiontree)'''

  with open('./model/decision.tree', 'rb') as decisiontree:
    decisiones = pickle.load(decisiontree)

  total_0 = (df_test['Y'] == 0).sum()
  total_1 = (df_test['Y'] == 1).sum()
  right_pred_0 = 0
  right_pred_1 = 0
  for i in range(len(df_test)):
    if int(df_test.iloc[i][str(df.columns[-1])]) == int(prediction(decisiones, df_test.iloc[i])):
      if int(df_test.iloc[i][str(df.columns[-1])]) == 0:
        right_pred_0 += 1
      elif int(df_test.iloc[i][str(df.columns[-1])]) == 1:
        right_pred_1 += 1
  print(right_pred_0/total_0, right_pred_1/total_1)

  print("Accuracy: {}%".format( ((right_pred_0 + right_pred_1) / (len(df_test) + 1)  ) * 100))