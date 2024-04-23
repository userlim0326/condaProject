import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import numpy as np

kn = KNeighborsClassifier()
figure = plt.figure()
data = pd.read_csv('Fish.csv')

bream_length = data.loc[data.Species == 'Bream', ['Length2']].iloc[:, 0].to_list()
bream_weight = data.loc[data.Species == 'Bream', ['Weight']].iloc[:, 0].to_list()

smelt_length = data.loc[data.Species == 'Smelt', ['Length2']].iloc[:, 0].to_list()
smelt_weight = data.loc[data.Species == 'Smelt', ['Weight']].iloc[:, 0].to_list()

# test set 작성
fish_length = bream_length + smelt_length
fish_weight = bream_weight + smelt_weight
# print(fish_length);print(fish_weight)
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]  # 2차원 list
fish_target = [1] * 35 + [0] * 14
print(fish_data)

# 1. Sampling bias(샘플링 편향)적인 train set 작성
train_input = fish_data[:35] # 리스트의 슬라이싱 마지막 인덱스는 미포함.
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]
kn = kn.fit(train_input, train_target)
score = kn.score(test_input, test_target)
print(score) # 0.0 출력 => 훈련세트와 테스트세트의 편향적인 구성으로 정확도가 매우 나쁨

# 2. Sampling unbias(샘플링 비편향)적인 train set 작성
np_fish_data = np.array(fish_data) # fish_data를 numpy로 변형
np_fish_target = np.array(fish_target)# fish_target를 numpy로 변형
# print(np_fish_data); print(np_fish_data.shape)

np.random.seed(42) # 고정된 난수를 설정
index = np.arange(49) # 0~ 48까지 배열 생성
np.random.shuffle(index) # 섞기
print(index)

# print(np_fish_data[[1,3]])  #넘파이는 슬라이싱외 배열인덱싱 제공(복수 데이터를 개별적 선택)
train_input = np_fish_data[index[:35]]
train_target = np_fish_target[index[:35]]
print(train_input)
print(np_fish_data[13], train_input[0])

test_input = np_fish_data[index[35:]]
test_target = np_fish_target[index[35:]]

# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(test_input[:,0], test_input[:,1])
# plt.xlabel("length")
# plt.ylabel("height")
# plt.show()

kn.fit(train_input, train_target) #데이터 일부만 훈련시킴
print(kn.score(test_input, test_target)) #나머지 데이터 적용하여 정확도 산출
print(kn.predict(test_input)) # test_input에 있는 데이터만 적용했을 때
print(test_target) # test_input에 타깃에 도미인지, 빙어인지 판별 값이 일치
print(test_input)
print(kn.predict([[25, 150]]))