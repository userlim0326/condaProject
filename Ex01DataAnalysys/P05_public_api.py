import pandas as pd
data = pd.read_csv('OBS_ASOS_TIM_2020.csv', encoding='cp949')
print(data)
print(data.dtypes)

print(f'{"데이터 전처리":=^20}')
data = data[['지점명', '일시', '기온(°C)']]
# print(data['지점명']=='서울')
is_seoul = data['지점명']=='서울'
seoul = data[is_seoul]
# print(seoul, len(seoul))
is_busan = data['지점명']=='부산'
busan = data[is_busan]
# print(busan, len(busan))

temp = []
for i in seoul['일시']:
  temp.append(i.split(' ')[0])
lack_seoul = []
unique_temp = set(temp)
for i in unique_temp:
  if temp.count(i) != 24:
    lack_seoul.append(i)
# print(lack_seoul)

temp = []
for i in busan['일시']:
  temp.append(i.split(' ')[0])
lack_busan = []
unique_temp = set(temp)
for i in unique_temp:
  if temp.count(i) != 24:
    lack_busan.append(i)
# print(lack_busan)

lack_data = list(set(lack_seoul+lack_busan))
for i in lack_data:
  seoul = seoul[seoul['일시'].str.split(' ').str[0] != i]
  busan = busan[busan['일시'].str.split(' ').str[0] != i]

print(len(seoul))
print(len(busan))

import numpy as np
time=['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00',
     '13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
zeros = np.zeros((24, 2))
df = pd.DataFrame(zeros, index=time, columns=['seoul', 'busan'])
print(df)

# for i in range(0,len(seoul),24):
#   print(seoul.iloc[i:i+24])

print(f'{"데이터 분석":=^20}')
max_idx = []
for i in range(0,len(seoul),24):
    max_idx.append(seoul.iloc[i:i+24,2].idxmax())
for i in max_idx:
    df.loc[seoul.loc[i][1].split(' ')[1]]['seoul'] +=1
max_idx = []
for i in range(0,len(busan),24):
    max_idx.append(busan.iloc[i:i+24,2].idxmax())
for i in max_idx:
    df.loc[busan.loc[i][1].split(' ')[1]]['busan'] +=1
print(df)
import matplotlib.pyplot as plt
# 겹친 바그래프
# plt.bar(df.index, df['seoul'], color='red')
# plt.bar(df.index, df['busan'], color='blue')
# plt.xticks(rotation=45)

plt.figure(figsize=(10,10))
# 가로바 그래프
# plt.barh(df.index, df['seoul'], color='red')
# plt.barh(df.index, df['busan'], color='blue')

# 바와 플롯 혼용 그래프
# plt.bar(df.index, df['seoul'], color='red')
# plt.plot(df.index, df['busan'], color='blue')

# 나란히 2개의 바그래프
x_range = np.arange(len(df))
plt.bar(x_range- 0.2, df['seoul'],width=0.4, color='red')
plt.bar(x_range+ 0.2, df['busan'],width=0.4, color='blue')

plt.xticks(rotation=45, ticks=x_range, labels=df.index)
plt.show()




