import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

knr = KNeighborsRegressor()
data = pd.read_csv('Fish.csv')

perch_length = data.loc[data.Species == 'Perch', ['Length2']].iloc[:, 0].to_list()
perch_weight = data.loc[data.Species == 'Perch', ['Weight']].iloc[:, 0].to_list()
train_input, test_input, train_target, test_target \
  = train_test_split(perch_length, perch_weight, random_state=42)
print(type(train_target), len(train_target))
train_input = np.array(train_input).reshape(-1,1)
test_input = np.array(test_input).reshape(-1,1)

knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.predict([[50]]))

distances, indexes = knr.kneighbors([[50]])

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.predict([[50]])) # length로부터 값을 예측
print("기울기 {}".format(lr.coef_))
print("절편 {}".format(lr.intercept_))

print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))
# input > test 과대 적합

print(type(train_input))
plt.scatter(train_input, train_target) # 훈련세트 산점도
plt.scatter(np.array(train_input)[indexes], np.array(train_target)[indexes], marker="D") # 이웃샘플 산점도
plt.plot([15, 50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_+ lr.intercept_])
plt.scatter(50, 1241, marker="^", color="r")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()




