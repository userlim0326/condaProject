import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn

figure = plt.figure()
data = pd.read_csv('Fish.csv')

bream_length = data.loc[:, 'Length2'].to_list()
# print(bream_length)

bream_length = data.loc[data.Species == 'Bream', ['Length2']]
# print(type(bream_length))  # bream_length는 DataFrame
# print(bream_length)
bream_length = bream_length.iloc[:, 0].to_list()
# print(bream_length)

bream_weight = data.loc[data.Species == 'Bream', ['Weight']].iloc[:, 0].to_list()
# print(bream_weight)
# plt.scatter(bream_length, bream_weight)
# plt.title('Bream')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

smelt_length = data.loc[data.Species == 'Smelt', ['Length2']].iloc[:, 0].to_list()
smelt_weight = data.loc[data.Species == 'Smelt', ['Weight']].iloc[:, 0].to_list()
# print(smelt_length, smelt_weight)

# 그래프를 겹칠 경우
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.title('Bream & Smelt')
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()

# 그래프를 하나의 figure에 띄울 경우
# ax1 = figure.add_subplot(121)
# ax1.scatter(bream_length, bream_weight)
# ax1.set_xlabel('length')
# ax1.set_ylabel('weight')
#
# ax2 = figure.add_subplot(122)
# ax2.scatter(smelt_length, smelt_weight, color="pink")
# ax2.set_xlabel('length')
# ax2.set_ylabel('weight')
# plt.show()

# 그래프를 각각의 창으로 여러개 띄울 경우
# plt.figure(1)
# plt.scatter(bream_length, bream_weight)
# plt.title('Bream')
# plt.xlabel('length')
# plt.ylabel('weight')
#
# plt.figure(2)
# plt.scatter(smelt_length, smelt_weight)
# plt.title('Smelt')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

length = bream_length + smelt_length
weight = bream_weight + smelt_weight
print(len(bream_length), len(smelt_length))

fish_data = [[l, w] for l, w in zip(length, weight)]  # 2차원 list
# fish_data = list(zip(length, weight)) # 1차원 list
# print(fish_data)

fish_target = [1] * 35 + [0] * 14
print(fish_target)

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)  # 학습
score = kn.score(fish_data, fish_target)  # 정확도
print("정확도: %.1f" % score)

l = 30; w = 600
# plt.scatter(l, w, marker="^", color="r")
# plt.show()
print(kn.predict([[l, w]]))
print(kn._fit_X)
print(kn._y)

kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)  # 학습
score = kn49.score(fish_data, fish_target)  # 정확도
print(score)
print(35/49)










