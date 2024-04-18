# raise Exception('강제 예외 발생')
# raise ValueError
error_message = """Traceback (most recent call last):
  File "C:\\workspace\\PycharmProjects\\pythonProject\\Ex01DataAnalysys\\P01_textdata_get.py", line 1, in <module>
    raise Exception('강제 예외 발생')
Exception: 강제 예외 발생
"""
# 파이썬의 기본 문법들을 활용하여 텍스트 데이터 내에서 불필요한 데이터를 제고하거나
# 패턴을 찾아 필요한 데이터만 분리하는 등의 데이터 분석을 진행하기 위함.

error_date = "[2024-04-11 14:00:00]"
print(error_message)
# f = open('2024-04-11_error_log.txt', 'w', encoding='utf8')
with open('2024-04-11_error_log.txt', 'w', encoding='utf8') as f:
  print(error_message, error_date , file=f)
f.close()

f = open('2024-04-11_error_log.txt', 'r', encoding='utf8')
data = f.read()
print(data)
f.close()

f = open('2024-04-11_data.csv', 'r', encoding='utf8')
data = f.read()
print(data)
f.close()

import csv
f = open('2024-04-11_data.csv', 'r', encoding='utf8')
data = csv.reader(f)
print(type(data))
for i in data:
  print(i)

import json
a = [1,2,3,{'4':5, '6':7}]
b = json.dumps(a) # 파이썬 객체 -> JSON 변환
print(b)
c = json.loads(b) # JSON -> 파이썬 객체 변환
print(c)

from bs4 import BeautifulSoup
from urllib.request import urlopen
html = urlopen('https://news.naver.com/section/105')
bs = BeautifulSoup(html.read(), 'html.parser')

# 1) css selector를 활용
titles = bs.select('.ct_scroll_wrapper>div>div>ul>li>a')
titles = bs.select('.as_headline>.sa_list>._SECTION_HEADLINE>.sa_item_inner>.sa_item_flex>.sa_text>.sa_text_title>strong')
print(len(titles))
for i in range(len(titles)):
    print(i+1, titles[i].text)

# find 찾으면 첫번째 꺼, find_all 복수개를 들고 옴
# first = bs.find('div', class_="as_section_headline").find('div', class_='as_headline').find('ul', class_='sa_list')
# lis = first.find_all('li', class_="_SECTION_HEADLINE")
# for s in lis:
#   temp = str(s.find('strong', class_="sa_text_strong"))
#   start = temp.find('">')+2
#   end = temp.find("</")
#   print(temp[start:end])

#정규 표현식(Regular Expression)
a="1,2,3,4,5"
for i in a:
  if i != ',': print(i, end=' ')
print()
print(a.replace(',', ' '))
print(a)
print(a[::2])

b = '''
  이길동, 01-1111-1111, 11, 0000
  홍길동, 02-111-2222, 21, 1234
  박길동, 051-0000-0000, 34, 4567
'''

print(b.strip())
print(b.strip().split('\n'))
for i in b.strip().split('\n'):
  i = i.split(',')[0].strip()
  print(i)

# https://m.blog.naver.com/cjinnnn/221329842667
# 정규표현식 : 특정한 규칙을 가진 문자열을 표현하기 위해 사용하는 형식(주로 검색, 치환에 활용)
# 1)  . : 1개 문자와 일치
# 2)  [] : 대괄호안의 문자중 하나 선택
#   ex) [abc]d -> ad, bd, cd 포함, [a-zA-Z]->알파벳 모두 포함, [가-힣],[0-9]
# 3)  * : 0개 이상의 문자 포함
#   ex) a*b -> b, ab, aab, aaaaaaab 등
# 4)  {m,n} : m회 이상 n회 이하
#   ex) a{1,3}b -> ab, aab, aaab
# 5)  그외
#   ?, +, (), ^ 등 다양한 규칙이 존재함

import re
a = "1,2,3,4,5"
pattern = re.compile('[^0-9]')  # ^0-9를 제외한 것
result = re.sub(pattern, '', a)
print(result)

pattern = re.compile('[^가-힣]')  # ^0-9를 제외한 것
result = re.sub(pattern, '', b)
print(result)

pattern = re.compile('/breakingnews/section/105/.*"')  # /breakingnews로 시작하는 주소들
result = pattern.search(str(bs))
print(result.group())

pattern = re.compile('[a-z:/.]*')  #
result = pattern.search(result.group())
print(result.group())

