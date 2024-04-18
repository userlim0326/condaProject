import pandas as pd

'''
행과 열로 이루어진 데이터를 쉽게 다룰 수 있도록 도와 주는
파이썬의 데이터 분석 전용 외부 모듈
특히 대용량의 데이터를 처리하는데 편리함
'''
# 1. Pandas의 자료형
#   1)Series 자료형
#       • 인덱스와 값을 가지고 있음
#       • 별도로 인덱스, 값을 출력할 수있음
#       • 정의할 때 인덱스를 따로 정해줄 수있음
#       • 파이썬의 딕셔너리 자료형
#       • Numpy의 Array 자료형도 Series 자료형으로 만들 수있음
#       • 딕셔너리의 키가 Series의 인덱스가 됨
sr1 = pd.Series([10, 20, 30, 40], index=['다현', '정연', '쯔위', '사나'])
sr2 = pd.Series([50, 60, 70, 80], index=['다현', '정연', '쯔위', '사나'])
sr3 = pd.Series([11, 22, 33, 44], index=['다현', '사나', '모모', '재남'])
print(sr1, type(sr1))
print(sr1.index, type(sr1.index))

sr12 = sr1 + sr2
print(sr12, '\n')

sr13 = sr1 + sr3
print(sr13)

a = pd.Series([1, 2, 3, 4])
print(a, type(a))
print("a.argmin():",a.argmin()) #최소값의 인덱스를 반환
print("a.argmax():", a.argmax())#최대값의 인덱스를 반환
print("a.mean():", a.mean()) #평균을 반환
print("a.unique():", a.unique()); # 중복값을 제외한 유니크한 값 반환

#   2)DataFrame 자료형
# • 행과 열로 이루어진 자료형
# • Series와 마찬가지로 파이썬의 딕셔너리 자료형 또는 Numpy의 array로도 정의할 수있음
#   (예) Series vs. DataFrame
# • Series: 인덱스, 값으로만 구성, 1차원 배열 형태의 자료 구조
# • DataFrame: 행과 열로 구성, 2차원 테이블 형태의 자료 구조

a = pd.Series({
  'a': [1, 1, 1],
  'b': [2, 2, 2],
  'c': [3, 3, 3]
})

b = pd.DataFrame({
  'a': [1, 1, 1],
  'b': [2, 2, 2],
  'c': [3, 3, 3]
})
print(a, type(a));
print(b, type(b))

a = pd.DataFrame({'a': (1, 2), 'b': 1, 'c': 3})  # 컬럼에 ()가 있으면 각각에 대한 인덱스 생성
print(a)
print(a.index);
print(a.columns)
a.index = ['x', 'y']
a.columns = ['i', 'j', 'k']
print("{0:=^40}".format(''))
print(a)
print(a.iloc[0])   # iloc: 행 인덱스로 값을 가져오기
print(a.loc['x'])  # loc:  행 이름으로 값을 가져오기

a = pd.DataFrame({'a':(1,2,3), 'b':[4,5,6], 'c':[7,8,9]})
print(a)
#    a  b  c
# 0  1  4  7
# 1  2  5  8
# 2  3  6  9

print(a.describe()) # DataFrame에서 계산 가능한 값들에 대하여 결과를 간략하게 표시
print(a.sum()) # default(axis=0)는 열 기준으로 합을 구함.
print(a.sum(axis=1)) #axis=1은 행 기준으로 합을 구함.
print(a.prod()) # default(axis=0)는 열 기준으로 곱을 구함.
print(a.prod(axis=1)) #axis=1은 행 기준으로 곱을 구함.
print(a.cumsum())  # 누적합에 대하여 연산
print(a.cumprod()) # 누적곱에 대하여 연산
print(a.min());print(a.min(axis=1));
print(a.max());print(a.max(axis=1));
print(a.mean());print(a.mean(axis=1)) # 평균
print(a.median());print(a.median(axis=1)) # 중간값
print(a.std());print(a.std(axis=1)) # 표준편차
print(a.var());print(a.var(axis=1)) # 분산

data = pd.read_csv('movies.csv')
print(data)

genre = []
for i in data['genres']:
  genre.extend(i.split('|'))
print(len(genre))
# print(genre)

np_genre = pd.Series(genre)
unique_genre = pd.unique(np_genre.sort_values())
print(len(unique_genre));print(unique_genre)

import numpy as np
zero_data = np.zeros(len(unique_genre))
result = pd.DataFrame(zero_data, index=unique_genre, columns=['count'])
print(result)

for i in genre:
  result.loc[i] += 1
  # print(result.loc[i])

print(result)

import matplotlib.pyplot as plt
x = np.array(unique_genre).tolist() # List로 변환
y = result["count"].values.tolist() # List로 변환
print(x); print(y)
# plt.bar(x,y) # 수직 바 그래프
# plt.barh(x,y) # 수평 바 그래프
plt.barh(x, result["count"].values.tolist(), color='#e35f62')
plt.yticks(x, np.array(unique_genre).tolist())
plt.show()










