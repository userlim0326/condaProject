# Decision Tree
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
print(wine.head()) #alcohol  sugar    pH(산도)  class(0:레드,1:화이트)
print(wine.info())
print(wine.describe())

import numpy as np
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = \
  train_test_split(data, target, random_state=42, test_size=0.2)
print(train_input.shape, test_input.shape)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
print(lr.coef_, lr.intercept_)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# plt.figure(figsize=(10,7))
# # plot_tree(dt)
# plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar','pH'])
# plt.show()

dt = DecisionTreeClassifier(max_depth=3 ,random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
# plt.figure(figsize=(20,15))
# plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar','pH'])
# plt.show()

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
# 결론: 결정 트리에서는 전처리 과정이 필요치 않다.

print(dt.feature_importances_) #특성에 대한 중요도 출력
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar','pH'])
plt.show()