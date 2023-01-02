# 보스턴 집값 예측

# 알고리즘: MLP (Multi-Layer Perceptron)
# 문제유형: 회귀
# 평가지표: 평균 제곱 오차
# 주요 패키지: torch, torch.nn, sklearn

# 파이토치 학습 과정
# 1. 모델 정의
# 2. 데이터 불러오기
# 3. 손실 계산
# 4. 오차 역전파 및 최적화 (가중치 업데이트)
# 5. 원하는 만큼 반복

# %%

from sklearn.datasets import load_boston

dataset = load_boston()
print(dataset.keys())

# dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename', 'data_module'])
# data : 특징값
# target : 정답
# feature_names : 각 특징의 이름
# DESCR : description 의 약자로 데이터셋에 대한 전반적인 정보
# filename : 데이타셋의 csv 파일이 존재하는 위치

# %%

# 1. 데이터 불러오기

import pandas as pd
from sklearn.datasets import load_boston
dataset = load_boston()
dataFrame = pd.DataFrame(dataset['data'])
dataFrame.columns = dataset['feature_names']
dataFrame['target'] = dataset['target']

print(dataFrame.head())

# %%

# 2. 모델 정의 및 학습하기

# 선형회귀 : 데이터를 y 와 x 의 관계를 나타내는 직선으로 나타내는 방법
# MSE(평균 제곱 오차) :오차에 제곱을 취하고 평균을 낸 값 --> 작은 오차와 큰 오차를 강하게 대비시킬 수 있어서 유용

import torch
import torch.nn as nn

from torch.optim.adam import Adam

# 1 - 모델 정의
model = nn.Sequential(
    nn.Linear(13, 100), # MLP 모델의 의미하며 13 입력차원, 100 출력 차원
    nn.ReLU(),
    nn.Linear(100, 1)
)

# 2 - 입력값, 정답 
X = dataFrame.iloc[:, :13].values # 정답을 제외한 특징을 x 에 입력
Y = dataFrame["target"].values # 데이터프레임의 tartget 값을 추출

batch_size = 100
lr = 0.001

# 3 - 가중치를 수정하는 최적화 함수 정의
# 최적화 기법은 역전파된 오차를 이용해 가중치를 수정하는 기법
optim = Adam(model.parameters(), lr = lr)

# 4 - 학습
for epoch in range(200):
    for i in range(len(X)//batch_size):
        start = i * batch_size
        end = start + batch_size

        x = torch.FloatTensor(X[start:end]) # 파이토치 텐서로 변환
        y = torch.FloatTensor(Y[start:end])

        optim.zero_grad() # 5 - 가중치의 기울기를 0으로 초기화
        preds = model(x) # 6 -모델의 예측값 계산
        loss = nn.MSELoss()(preds, y) # 7 - MSE 손실 계산
        loss.backward() # 8 - 오차 역전파
        optim.step() # 9 - 최적화 진행

    if epoch % 20 == 0:
        print(f"epoch{epoch} loss:{loss.item()}")

# %%

# 3. 모델 성능 평가
prediction = model(torch.FloatTensor(X[0, :13]))
real = Y[0]
print(f"예측 값:{prediction.item()} 정답:{real}")





# %%
