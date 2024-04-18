# 변수의 명명규칙
# 1) 예약어 안됨
# 2) _, 영문자(대소문자 구별), 숫자(시작 안됨)
# 3) 특수문자, 공백 안됨
# 4) 클래스는 Pascal case, 변수나 함수는 Snake case
# 5) Python에서는 null 대신 None 사용
'''
[ 자료형 ]
숫자 자료형 : int, float, complex
불 자료형 : True, False
군집 자료형 : str, list, tuple, dict, set
동적 자료형 : 입력시 타입이 결정
'''
import random

# 변수는 동적 할당(선언 + 초기화)
a = 10
b = True
c = 3.14
d = 'hello'
e = 'c'
f = {"name", "age", "email"}
g = 10 + 3j + 6J # complex(복소수) 파이썬에서는 imaginary number의 약자 j, J 사용
h = complex(3, -4) # 실수부, 허수부
print(h.real)
print(h.imag)
print(type(a), type(b), type(c), type(d), type(e), type(f), type(g), type(h))

# k = (a = 10 + 20) 할당문이 할당문을 포함할 수 없다.
a = b= 10
a, b = 1, 2
print(a, b)
a, b = b , a
# 일반적인 변수의 교환 tmp = a; a = b; b = tmp
print(a, b)

del a, b
# print("del 이후", a, b) # NameError 발생

print("{0:=^20}".format('변수의 종류'))
print("{0:=^20}".format('1. null :: NoneType'))
print(type(None))

print("{0:=^20}".format('2. 숫자형 :: int, float, complex'))
a = 10; print(a); print(type(a))
a = 12.345; print(a); print(type(a), end="\t")
a = 0b1010; print(a); print(type(a), end="\t")
a = 0o12; print(a); print(type(a), end="\t")
a = 0x2a; print(a); print(type(a), end="\t")
a = 123e2; print(a); print(type(a), end="\t")
a = 123E-2; print(a); print(type(a), end="\t")
print(format(10, 'b'))
print(format(10, 'o'))
print(format(10, 'x'))
print(format(10, 'd'))
print(format(10, '#b'))
print(format(10, '#o'))
print(format(10, '#x'))
print(format(10, '#X'))
print(format(10, '#d'))

a = 1234.5678; print(int(a)) #형변환
a = 10; print(float(a)) #형변환
# print(int(10 + 5j))
try:
  a = "123a"; print(type(int(a)))
except:
  print("문자열이 포함 되어 있습니다.")

a = 10 + complex(3); print(type(a)) #complex
a = 123456789123456789; print(type(a)) # python 3버전부터는 long이  int로 변경되었음

# 난수 발생
a= random.random()* 45; print(int(a))
a = random.uniform(1, 100); print(int(a))

print("{0:=^20}".format('3. 논리형 :: bool'))
print(type(10>5))
print((1>2) or (2<5))
print(type(False))
print("{0:=^20}".format('Truthy'))
print(bool(-1))
print(bool(1000))
print(bool(' '))

print("{0:=^20}".format('Falsy'))
print(bool(None))
print(bool(0))
print(bool(''))

print("{0:=^20}".format('4. 문자형 :: str, chr'))
c = chr(48); print(type(c))
print(type(chr(97)))
print(ord('A'))
print("'A' == chr(65):", 'A' == chr(65))
print("'A' == ord('A'):", 'A' == ord('A'))

# python은 상수가 없다. python은 동적 언어이기 때문에 상수에 대한 키워드가 없다.
from typing import Final
SIZE:Final = 5
SIZE = 10
print(SIZE) # 사용가능. 재할당을 막을 수는 없다.

# class를 만들어서 상수처럼 사용 가능
import common.constant as const

const.PI = 3.14
print(const.PI)

# const.PI = 3.141592 # 재할당 할 경우 에러 발생
# print(const.PI)