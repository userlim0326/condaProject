import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

knr = KNeighborsRegressor()
data = pd.read_csv('Fish.csv')
# print(type(data)) #<class 'pandas.core.frame.DataFrame'>

# perch_full = Length2, Height, Width
perch_full = data.loc[data.Species == 'Perch', ['Length2', 'Height', 'Width']].to_numpy()
perch_weight = data.loc[data.Species == 'Perch', ['Weight']].iloc[:, 0].to_list()

train_input, test_input, train_target, test_target \
  = train_test_split(perch_full, perch_weight, random_state=42)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(include_bias=False)
# poly.fit([[2,3]]); print(poly.transform([[2,3]]))
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
# print(train_poly.shape); print(train_input.shape)
# print(poly.get_feature_names_out())

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
# 훈련세트에 대하여 결론 속성을 늘릴수록 완벽하게 학습을 할 수 있으나
# 테스트세트에서 형편없는 점수를 받기에 훈련세트에 대하여 너무 과대적합되었다고 본다.

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()  # 평균과 표준편차를 직접 구해 특성을 표준점수로 변경
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 규제 추가를 위한 2가지 모델 : Ridge, Lasso

# Ridge 모델 적용
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
# 비교적 좋은 성능.

# 릿지,라쏘 사용할 때 규제의 양을 임의로 조절 위해 alpha 매개변수 설정하기
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  ridge = Ridge(alpha=alpha)
  ridge.fit(train_scaled, train_target)
  train_score.append(ridge.score(train_scaled, train_target))
  test_score.append(ridge.score(test_scaled, test_target))

# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.show()
# 그래프에서 훈련데이터의 그래프와 테스트데이터의 그래프가 가장 가깝게 만나는 곳은 -1이다
# 따라서 alpha 값은 0.1로 하여 적용한다.
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  lasso = Lasso(alpha=alpha, max_iter=20000)
  lasso.fit(train_scaled, train_target)
  train_score.append(lasso.score(train_scaled, train_target))
  test_score.append(lasso.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print("{0:=^20}".format('결과'))
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
# 결론 : 알파값을 조정한 후에 훈련데이터는 99퍼센트에 가까워졌고, 테스트와도
# 적당한 과대적합을 이루었다.