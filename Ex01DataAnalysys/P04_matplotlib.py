'''
데이터 시각화: 데이터 분석 결과를 시각적으로 표현하고 전달하는 과정
전문지식이 없어도 누구나 쉽게 데이터를 인지하고 활용할 수 있다.
시각화는 주로 그래프를 사용
axe는 축, 회전축
'''
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

# 한글 지원
from matplotlib import font_manager, rc
font_path = "C:\\Windows\\Fonts\\HANBaekM.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

'''
figure = plt.figure()
#add_subplot([총 행의 수][총 열의 수][subplot의 인덱스(1부터 시작)])
# axes1 = figure.add_subplot(121)  # 1행 2열에서 1열
# axes2 = figure.add_subplot(122)  # 1행 2열에서 2열
axes1 = figure.add_subplot(221)  # 2행 2열에서 1열
axes2 = figure.add_subplot(222)  # 2행 2열에서 2열
axes3 = figure.add_subplot(223)  # 2행 2열에서 3열
axes4 = figure.add_subplot(224)  # 2행 2열에서 4열
plt.show()
'''

# 직선, 꺽은 그래프
'''
figure = plt.figure()
axes = figure.add_subplot()
x = [1,2,3,4]
y = [2,4,6,8]
axes.plot(x,y)

axes = figure.add_subplot(111)
x = [1,2,3,4]
y = [2,4,6,8]
x2 = [1,2,3,4]
y2 = [4,1,3,6]
axes.plot(x,y, color="red", linestyle='dashed', marker="v", label="x")
axes.plot(x2, y2, color="k", linestyle="dotted", marker="o", label="y")
axes.legend() # plot 속성의 label을 가지고 생성
axes.set_title("x and y")
axes.set_ylabel("y")
axes.set_xlabel("x")
plt.show()
'''

# 막대그래프
'''
figure = plt.figure()
ax = figure.add_subplot(111)
x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Bar graph')
bar = ax.bar(x,y, label="y", color="pink")
ax.legend()
ax.set_ylim(0,10)
# 세로에서 숫자 넣기
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height,
             ha='center', va='bottom', size = 12)
plt.show()
'''

# 복합 축(bar and plot)
'''
figure = plt.figure()
axes = figure.add_subplot(111)
axes2 = axes.twinx()

x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

x2 = [1, 2, 3, 4]
y2 = [4, 4, 3, 6]

axes.bar(x,y,color='green', label='bar')
axes2.plot(x2,y2,color='r', label='plot')

axes.legend()
axes2.legend(loc=1)
plt.show()
'''

# Scatter (산점도) graph
'''
figure = plt.figure()
axes = figure.add_subplot(111)
x = [1, 2, 3, 4]
y = [2, 4, 6, 8]
x2 = [1, 1, 3, 4]
y2 = [6, 2, 4, 6]

axes.scatter(x,y)
axes.scatter(x2,y2)
plt.show()
'''

# Pie (원) graph
'''
figure = plt.figure()
axes = figure.add_subplot(111)
ratio = [1,1,2,3]
label = ['하나','b','c','d']
# ratio = [34,32, 16, 18]
# label = ['Apple', 'Banana', 'Melon', 'Grapes']
axes.pie(ratio, labels=label, autopct='%.1f%%')
plt.savefig('pie_graph')
plt.show()
'''

'''
# 이미지 파일을 읽어서 출력하기
image = img.imread('pie_graph.png') # 이미지 불러옴
plt.imshow(image) # 이미지 그래프 설정
plt.show() # 그래프 출력
'''

# Wordcloud
# import wordcloud
# file = open('노무현_한·일관계에 대한 특별 담화문.txt', 'r', encoding='utf8')
# data = file.read()
# data = data.replace('\n','')
# data = data.replace('일본이','')
# data = data.replace('대한','')
# data = data.replace('일본의','')
# data = data.replace('일본은','')
# data = data.replace('일본','')
# data = data.replace('것입니다.','')
# wc = wordcloud.WordCloud(font_path='C:\\Windows\\Fonts\\HANBaekM.ttf', background_color='white',
#                          stopwords=['있는', '우리는', '한일', '결코'])
# wc.generate_from_text(data)
# wc.to_image()
# wc.to_file('Roh.jpg')





