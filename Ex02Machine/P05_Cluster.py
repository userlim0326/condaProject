# Cluster Algorithm (군집 알고리즘: 비지도 학습)
# 비슷한 샘플끼리만 모으기
import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')
print(fruits.shape)
print(fruits[0, 0, :])  # 첫번째 이미지의 첫행을 출력
# plt.imshow(fruits[0], cmap='gray') # 사진으로 찍은 이미지를 넘파이 배열로 변환할 때 반전한 이미지
# plt.show()

# plt.imshow(fruits[0], cmap='gray_r')
# plt.show()

# fig, axs = plt.subplots(1,2)
# axs[0].imshow(fruits[100], cmap='gray_r')
# axs[1].imshow(fruits[200], cmap='gray_r')
# plt.show()

apple = fruits[0:100].reshape(-1, 100 * 100)
pineapple = fruits[100:200].reshape(-1, 100 * 100)
banana = fruits[200:300].reshape(-1, 100 * 100)

print(apple.shape)
print(apple.mean(axis=1))

# histogram으로 각각의 이미지 평균값 보기
# plt.hist(np.mean(apple, axis=1), alpha=0.8)
# plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
# plt.hist(np.mean(banana, axis=1), alpha=0.8)
# plt.legend(['apple', 'pineapple', 'banana'])
# plt.show()

# bar로 각각의 이미지 평균값 보기
# fig, axs = plt.subplots(1,3, figsize=(20,5))
# axs[0].bar(range(10000), np.mean(apple, axis=0))
# axs[1].bar(range(10000), np.mean(pineapple, axis=0))
# axs[2].bar(range(10000), np.mean(banana, axis=0))
# plt.show()

# 이미지로 각각의 이미지 평균값 보기
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)
# fig, axs = plt.subplots(1,3, figsize=(20,5))
# axs[0].imshow(apple_mean, cmap='gray_r')
# axs[1].imshow(pineapple_mean, cmap='gray_r')
# axs[2].imshow(banana_mean, cmap='gray_r')
# plt.show()

# 사과 사진의 평균값이 apple_mean과 가장 가까운 사진 고르기를 위해 절대값 오차를 적용
# fruits 배열에 있는 모든 샘플에서 apple_mean을 뺀 절대값의 평균을 계산
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)
apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[apple_index[i * 10 + j]], cmap='gray_r')
    axs[i, j].axis('off')
plt.show()
