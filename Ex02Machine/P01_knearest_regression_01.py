import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import common.constant as const
import numpy as np

#1. perch_length, perch_weight의 list를 구하시오
#2. perch_length, perch_weight의 길이를 구하시오.
#3. 산점도를 출력하시오.

knr = KNeighborsRegressor()
data = pd.read_csv('Fish.csv')

perch_length = data.loc[data.Species == 'Perch', ['Length2']].iloc[:, 0].to_list()
perch_weight = data.loc[data.Species == 'Perch', ['Weight']].iloc[:, 0].to_list()
const.perches = len(perch_length)
print("길이: %d" % const.perches) # length, weight 각각 56개 출력
print(perch_length)
print(perch_weight)
# print(perch_weight)

# plt.scatter(perch_length, perch_weight)
# plt.xlabel("length")
# plt.ylabel("height")
# plt.title("Perch")
# plt.show()

# 농어의 길이(입력:input)를 통해서 몸무게(결과:target)를 예측해 보자
# train_input : 농어의 길이, train_target : 농어의 무게
train_input, test_input, train_target, test_target \
  = train_test_split(perch_length, perch_weight, random_state=42)
print(type(train_target), len(train_target))
# print(train_target)

# example: 2차원 배열로 변경
# test_array = np.array([1,2,3,4])
# print(test_array.shape, test_array)
# test_array = test_array.reshape(2,2)
# print(test_array.shape, test_array)
print(train_input)
print(test_input)

# reshape(-1, 1) 크기가 -1일 때 원소의 크기 만큼 채워라
train_input = np.array(train_input).reshape(-1,1)
test_input = np.array(test_input).reshape(-1,1)
print(train_input.shape, test_input.shape)
print(train_input);print(train_target)

knr.fit(train_input, train_target)
# 결정계수가 정확한 숫자를 맞히는 것은 불가능, 예측하는 값이나 타깃 모두 임의의 수치이기 때문.
print(knr.score(test_input, test_target)) # 테스트 셋 정확도: 0.992809406101064
print(knr.score(train_input, train_target)) # 훈련 셋 정확도: 0.9698823289099254
# 훈련셋 > 테스트셋 => 과대 적합
# 훈련셋 < 테스트셋 => 과소 적합
# 결론 농어의 훈련셋과 테스트셋은 과소 적합.

# 이유를 찾기 위해서 평균에 대한 오차를 출력해 보자.
from sklearn.metrics import mean_absolute_error
test_prediction = knr.predict(test_input) #테스트(length) 세트에 대한 예측
print("test_prediction: ", test_prediction) #예측된 값
print("test_target: ", test_target)
mae = mean_absolute_error(test_target, test_prediction) #예측된 값과 실제 값
print(mae) # 예측이 평균적으로 19.157142857142862g 정도 타깃과 다르다는 것을 알 수 있음.

# 그래서 근접하는 이웃의 갯수를 줄여 보자
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))

# 최종결론 과대 적합일경우 모델을 덜 복잡하게 만들어야 한다. 근접 갯수를 늘림
# 과소 적합일 경우 모델을 더 복잡하게 만들어야 한다. 근접갯수를 줄임.