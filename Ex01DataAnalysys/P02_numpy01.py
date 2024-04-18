import numpy as np
'''
Numpy : 대규모 다차원 배열을 쉽게 처리 할 수 있도록 도와 주는 파이썬의 외부 모듈
기본적으로 array라는 자료형으로 사용, 행렬 개념과 비슷
'''
# 1차원 넘파이 배열
a = [1,2,3,4,5]
print(a)
print(type(a))
b = np.array(a)
print(b)
print(type(b))
# 리스트: 요소를 기준으로 인덱싱, 넘파이행렬: 위치를 기준으로 인덱싱
print(a[0])
print(b[0])
print(a[0:3])
print(b[0:3])

'''
행렬 형태의 행렬 연산 지원
▪ 리스트
  •-, / 연산불가능
  • + 연결, * 반복연산만가능
▪ Numpy: 사칙연산모두가능
'''
c = np.array(a)
print(c, type(c))
print(a+c) # list + ndarray
print(c+c) # ndarray + ndarray
print(a*2)
print(c*2)
# print(a/2) list
# print(a-a) list

print(c+2)
print(c/2)
print(c-c)

print(np.sqrt(4))
print(np.log(10))
print(np.max([1,2,3])) # 일차원


