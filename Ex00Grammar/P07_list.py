import common.constant as const

def pArr(arr):
  for i in range(0, len(arr), 1):
    for j in range(0, len(arr[i]), 1):
      print(arr[i][j], end=" ")
    print(end="\n")

###################### 1차원 리스트 #######################
# array[값] 값은 인덱스가 되어 해당하는 원소를 추출
# array[:] array의 모든 성분을 추출
# array[값1:값2] 인덱스 값1에서 값2 사이의 원소를 추출
# array[값1:값2]=값3 해당값에 새로운 값3를 입력
arr = [1,2,3,4,5,6,7,8,9,10]
print(">>>",arr);print(arr[9]);print(arr[:]);print(arr[1:3])
arr[5:8] = 12  # 숫자를 작은 것에서 큰순으로 넣을 것
print(arr)

l = list();
l.append(1);  # 끝에 추가
l.insert(3, 2);  # 원하는 곳에 추가
l.extend([3, 4, 5])  # list를 추가
l = l + [6, 7]
del l[len(l) - 1];
print(l)  # 마지막 인덱스를 지움
del l[l.index(6)];
print(l)  # index(원소) 해당 원소가 있는 위치를 인덱스로 반환
l.remove(5)  # 4라는 원소를 찾아서 지움
print(l, id(l))  # 연산자로 더하기

l = [5, 6, 7];
print(l, id(l))
l.clear()  # 모두 삭제
for i in range(0, 10, 1):
  l.append(i)
print(l);
print(l[0]);
print(l[0:3])
l.append("hello")
l.append(True)
l.append(12.34)
lcopy = l.copy()  # 새로운 주소의 리스트를 생성, 주소가 다름
print(lcopy, id(lcopy))

listStr = ["Life", "is", "Good", "and", "Happy"]
print(listStr)
listStr.reverse(); print(listStr)
print(listStr.index("Good"))


###################### 2차원 리스트 #######################
import random
const.SIZE = 5 # 상수 선언

# 2차원 리스트 배열 만드는 방법  : 잘 사용 안함
npArr2 = []
for _ in range(const.SIZE):
  tmpArr = []
  for _ in range(const.SIZE):
    tmpArr.append(_)
  npArr2.append(tmpArr)
print(npArr2)

# 2차원 리스트 배열 만드는 방법
pythonList = [[random.randint(0,255) for _ in range(const.SIZE)] for _ in range(const.SIZE)]
print(pythonList)

lists = [
  ["이길동", 95, 85, 100],
  ["홍길동", 85, 95, 100],
  ["박길동", 100, 95, 85]
]
print(lists, type(lists))
print(lists[1], lists[0][0])

a = [[1,2,3], [4,5,6], [7,8,9]]
print(a[ :,2])
print(a[ :,2])
print(a[1,:2])

# array[행, 열] : 단일차원과 달리 다차원에서는 콤마(,)를 기준으로 행과 열로 분리
# array[행 시작,행끝 : 열시작,열끝], array[행 시작:행끝 , 열시작:열끝]
# https://m.blog.naver.com/hsj2864/220831822625

# 새로운 리스트 생성
lists2 = [[0 for j in range(len(lists[0])+1)] for i in range(len(lists))]
pArr(lists2)

for i in range(0, len(lists)):
  for j in range(0, len(lists[i]), 1):
    lists2[i][j] = lists[i][j]
    if j!= 0:
      lists2[i][len(lists[0])] += lists[i][j]
  print(end="\n")

lists3 = list()
for i in range(0, len(lists)):
  tmpList = list()
  for j in range(0, len(lists[i]), 1):
    tmpList.append(lists[i][j]);
    if j>1:
      tmpList.append(lists[i][j])
      tmpList.insert(i,lists[i][j-1] + lists[i][j])
  lists3.append(tmpList)

print("{0:=^20}".format('결과'))
pArr(lists2)
pArr(lists3)




