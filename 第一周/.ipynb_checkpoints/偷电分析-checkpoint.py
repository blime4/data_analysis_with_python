# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

###周期性分析###
power=pd.read_csv('./example/Ele_pow.csv',index_col=0)
print(power.head())
power.plot()

# +
###缺失值处理###
from scipy.interpolate import lagrange #导入拉格朗日插值函数

inputfile = './example/missing_data.xls' #输入数据路径,需要使用Excel格式；
outputfile = './example/missing_data_processed.xls' #输出数据路径,需要使用Excel格式

data = pd.read_excel(inputfile, header=None) #读入数据
data


# -

#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
  y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))] #取数
  y = y[y.notnull()] #剔除空值
  return lagrange(y.index, list(y))(n) #插值并返回插值结果



for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]: #如果为空即插值。
              data[i][j] = ployinterp_column(data[i], j)

data.to_excel(outputfile, header=None, index=False) #输出结果

data


###构建模型####
def cm_plot(y, yp):
  
  from sklearn.metrics import confusion_matrix

  cm = confusion_matrix(y, yp) 
  
  import matplotlib.pyplot as plt 
  plt.matshow(cm, cmap=plt.cm.Greens) 
  plt.colorbar() 
  
  for x in range(len(cm)): 
    for y in range(len(cm)):
      plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
  
  plt.ylabel('True label') 
  plt.xlabel('Predicted label') 
  return plt


