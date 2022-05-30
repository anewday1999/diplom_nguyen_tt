import numpy as np
import pandas as pd
def printTree(node, level=0):
  if node != None:
    if type(node) != np.int64 and type(node) != np.float64:
      printTree(node.right, level + 1)
      print(' ' * 21 * level + '-> ' + str(node.condition))
      printTree(node.left, level + 1)
    else:
      print(' ' * 21 * level + '-> ' + str(node))
def oversample(df):
  classes = df.Y.value_counts().to_dict()
  most_value = max(classes.values())
  classes_list = []
  for cl in classes:
      classes_list.append(df[df['Y'] == cl]) 
  classes_sample = []
  for i in range(1,len(classes_list)):
      classes_sample.append(classes_list[i].sample(most_value, replace = True))
  df_maybe = pd.concat(classes_sample)
  result = pd.concat([df_maybe, classes_list[0]], axis=0)
  result = result.reset_index(drop = True)
  return result