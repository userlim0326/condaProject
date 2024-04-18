'''
  Python의 연산자에 대하여 알아 봅시다.
'''

print("{0:=^20}".format('1. 사칙연산자'))
print("10을 3으로 나눴을 때 몫과 나머지를 구하시오.")
print("10을 3으로 나눴을 때 몫은 {} 나머지는 {}".format(10//3, 10%3))
print("10의 3승: {}".format(10 ** 3))
print("{0:=^20}".format('2. 비트연산자'))
print(0b1010 & 0b1100)
print(bin(0b1010 & 0b1100))
print(bin(0b1010 | 0b1100))
print(bin(0b1010 ^ 0b1100))
print(bin(~0b1010)) #2의 보수로 출력
print("%s<<2 = %s" %(bin(2), bin(2<<2))) # shift 연산자 >>, <<

print("{0:=^20}".format('3. 대입연산자'))
tot = 0
for i in range(1, 10, 2):
  tot += i
print("합은 %d" % tot)

print("{0:=^20}".format('4. 비교연산자 (>, <, <=, >=, ==, !='))
print("{0:=^20}".format('5. 논리연산자 (and, or, not)'))
print(not((1>2) and (4>3)) )

print("{0:=^20}".format('6. 삼항 연산자가 없다. 대신에'))
a = 1
a = 10 if a > 5 else 5
print(a)

print("{0:=^20}".format('7. 문자열 연산자 :: +, *'))
a = "hello"
ㄱ = "world"
print( a + ㄱ)
# print(10 + a) # 문자열은 같은 문자열일 경우 연산가능
print(str(10) + a)
a='A';
print("ascii(%s), str(%s) of Ascii Code is %d" % (ascii(a), str(a), ord('A')))
a=65;
# chr()는 매개변수가 숫자여야만 함.
print("ascii(%s), str(%s), chr(%s) of Ascii Code is %d" % (ascii(a), str(a), chr(a), ord('A')))
print("=" * 10)
print("Hello Python"[0])
print("Hello Python"[-2])
print("Hello Python"[0:12:2])