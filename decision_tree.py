#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:15:43 2020

@author: Manting
"""

def maxdepth():
    y_eff = []
    for i in range(10): 
        dtree = DecisionTreeClassifier(criterion='entropy',max_depth = i+1,
                                   random_state = 30,splitter = "random").fit(X_train, y_train)
        score = dtree.score(X_test,y_test)
        y_eff.append(score)
    print("max_depth = %s , score = %s" % (y_eff.index(max(y_eff)) + 1 , max(y_eff)))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score
from sklearn import tree

# 讀資料
df = pd.read_csv('character-deaths.csv')

# 資料前置處理
df.loc[df["Death Year"] >= 0, 'Death Year'] = 1 # 有數值的轉成1
df.fillna(0, inplace = True) # 空直補0
df = df.join(pd.get_dummies(df['Allegiances'])) # 將Allegiances轉成dummy特徵
y = df["Death Year"]
X = df.iloc[ : , 5 : ]

# 亂數拆成訓練集(75%)與測試集(25%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, train_size = 0.75)
print(X_test,y_test)

# max_depth參數測試
maxdepth()

# DecisionTreeClassifier進行預測
dtree = DecisionTreeClassifier(criterion='entropy',max_depth = 8,random_state = 0).fit(X_train, y_train)


# 產出決策樹的圖
fn = list(X.columns) 
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,3), dpi=200)
tree.plot_tree(dtree,
               feature_names = fn,
               filled = True)
fig.savefig('decision_tree.png',bbox_inches = 'tight')

# 計算數值
predictions = dtree.predict(X_test) 
matrix = confusion_matrix(y_test,predictions) 
precision_score = precision_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print(classification_report(y_test, predictions))
print("matrix:\n", matrix)
print("predictions:",precision_score)
print("accuracy:", accuracy)
print("recall:", recall)

# 輸出特徵
#print(dtree.tree_.feature)
#print(dtree.feature_importances_)

