import numpy as np

ary = np.array([[-1, -2], [3, 4]])
print(ary)

print(np.abs(ary))
print(np.sign(ary))  # 부호 1이면 양수, -1이면 음수
# print(np.sqrt(ary)) #음수의 루트값은 nan을 리턴
print(np.logical_not(ary))

print("🚀" * 15)
ary1 = np.array([[1, 2], [3, 4]])
ary2 = np.array([[5, 6], [7, 8]])
print(ary1 + ary2)
print(np.add(ary1, ary2))
print(np.subtract(ary2, ary1))
print(ary1 * ary2)
print(ary2 // ary1)  # 몫
print(ary2 % ary1)  # 나머지
print("🚩" * 15)
ary1 = np.random.randint(0, 255, size=(3, 3))
print(ary1)
ary2 = np.random.randint(0, 255, size=(3, 3))
print(ary2)
print(np.maximum(ary1, ary2))
print(np.minimum(ary1, ary2))
print(np.greater_equal(ary1, ary2))
print(np.equal(ary1, ary2))
print(np.power(ary1, ary2))
print(182 ** 186, type(182 ** 186))
print("▶" * 20)
ary1 = np.random.randint(0, 255, size=(3, 3))
print(ary1)
ary2 = np.random.randint(0, 255, size=(3, 3))
print(ary2)
condAry = np.random.choice([True, False], size=(3, 3))
print(condAry)
print(np.where(condAry, ary1, ary2))  # True면 ary1, False면 ary2
ary = np.random.randint(-128, 127, size=(3, 3))
print(ary)
print(np.where(ary < 0, 0, ary))
print("🐱‍"*15)
ary2 = np.random.randint(0, 255, size=(3,3))
print(ary2)
print(ary2.sum())
print(ary2.sum(axis=0)) # 세로의 합
print(ary2.sum(axis=1)) # 가로의 합

ary3 = np.zeros((4,4),int)
print(ary3)
for i in range(len(ary2)):
  for j in range(len(ary2[i])):
    ary3[i,j] = ary2[i,j]
    ary3[len(ary2), j] += ary2[i,j]
    ary3[i, len(ary2)] += ary2[i,j]
    ary3[len(ary2), len(ary2)] += ary2[i,j]
print(ary3)

ary2 = np.random.randint(0,255, size=(3,3))
print(ary2)
print(ary2.min())
print(ary2.max())
print(np.sort(ary2, axis=0)[::-1])
ary2 = np.sort(ary2.reshape(ary2.size))
print(ary2.reshape(3,3))
print("❤"*15)
ary = np.random.randint(0,9, size=10)
print(np.unique(ary))
ary1 = np.random.randint(0,9, size=10)
ary2 = np.random.randint(0,9, size=10)
print(ary1)
print(ary2)
print(np.intersect1d(ary1, ary2))
