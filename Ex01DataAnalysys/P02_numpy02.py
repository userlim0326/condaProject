import numpy as np
import random
import common.constant as const

const.SIZE = 5 # 상수 선언

## 넘파이 2차원 배열 생성
SIZE = 5  # 원본 크기
value = 1
myAry1 = np.arange(value, value+(SIZE*SIZE), 1)
myAry1 = myAry1.reshape(SIZE, SIZE)
print(myAry1)

# 2차원 넘파이 배열 만드는 방법 1
npArr = np.random.randint(0, 256, size=(const.SIZE, const.SIZE))
print(npArr)

npArr += 1
print(npArr)

tot = 0; cnt = 0
for i in range(len(npArr)):
  for j in range(len(npArr[i])):
    tot += npArr[i][j]; cnt += 1
print(f'Total: {tot} / Average: {tot/cnt}')

for i in range(len(npArr)):
  [print("%3d" % npArr[i][k], end=' ') for k in range(len(npArr[i]))]
  print()
print()

### 넘파이 배열의 인덱싱 ###
# array[행, 열] : 단일차원과 달리 다차원에서는 콤마(,)를 기준으로 행과 열로 분리
# array[행 시작,행끝 : 열시작,열끝], array[행 시작:행끝 , 열시작:열끝]
# https://m.blog.naver.com/hsj2864/220831822625
a2 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
print(a2[2,:]) # 2행만
print(a2[2:])  # 2행부터 끝행까지
print(a2[1:3,])  # 1~2행 추출
print(a2[:,3]) # 열의 인덱스가 3인경우만 추출
print(a2[:3,]) # 열의 인덱스가 3인경우만 추출
a2[:2, 1:3]=0
print(a2)
print(a2[3,2])
print(np.max(a2)) # 다차원

pythonList = [ random.randint(0,255) for _ in range(5)]
print('* 파이썬 리스트 --> ', pythonList)

numpyAry1 = np.array(pythonList)
print('* array(pythonList) --> ', numpyAry1)

numpyAry2 = np.arange(5)
print('* arange(5) --> ', numpyAry2)
numpyAry3 = np.arange(3, 8)
print('* arange(3, 8) --> ', numpyAry3)
numpyAry3 = np.arange(0, 100, 20)
print('* arange(0, 100, 20) --> ', numpyAry3)

numpyAry4 = np.ones(5)
print('* ones(5) --> \n', numpyAry4)
numpyAry5 = np.ones((3,4))
print('* ones((3,4)) )--> \n', numpyAry5)

numpyAry6 = np.zeros(5)
print('* zeros(5)--> ', numpyAry6)

numpyAry7 = np.empty(6)
print('* empty(6)--> ', numpyAry7)

numpyAry8 = np.full(5, 33)      # 1차원
numpyAry8 = np.full((5,3), 33)  # 2차원
print('* full(5, 33) --> ', numpyAry8)

numpyAry9 = np.identity(5)
print('* identity(5)--> \n', numpyAry9)