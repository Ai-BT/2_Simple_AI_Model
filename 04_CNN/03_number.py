# # 손글씨 분류

# 알고리즘: MLP (Multi-Layer Perceptron)
# 신경망의 출력을 그대로 사용하여, 확률 분포(소프트맥스)로 바꿔서 사용

# %%
# 1. 데이터 살펴보기

import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

# <학습 데이터, 테스트 데이터 분류>
# train=True, train=Flase 의 의미는 학습용 데이터를 불러오려면 True // 테스트용 데이터를 불러오려면 False
train_data = MNIST(root='./', train=True, download=True, transform=ToTensor())
test_data = MNIST(root='./', train=False, download=True, transform=ToTensor())

print(len(train_data))
print(len(test_data))

# %%

# <샘플 이미지 9개 출력>
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_data.data[i])
plt.show()

# %%

# 2. 데이터 불러오기 (Dataloader() 함수)

from torch.utils.data.dataloader import DataLoader

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# DataLoader 함수를 사용하면 우리가 원하는 배치크기, 셔플 여부를 설정 가능
# 학습용 데이터를 섞지 않고 학습하면 하나의 루틴에서 계속 학습 될 수 있습니다.
# 그래서 이미지의 순석를 섞어서 학습하는 것이 좋습니다.
# 테스트 데이터는 학습이 된 모델을 이용해서 순서대로 예측값과 정답을 비교하는 과정이므로
# 데이터를 섞을 필요는 없습니다.

# %%

# 3. 모델 정의 및 학습하기
# 모델 정의
# 데이터 호출
# 손실 계산
# 오차 역전파 및 최적화 (가중치 업데이트)
# 원하는 만큼 반복

# 기존의 학습과는 다르게, 이미지는 가로축과 세로축으로 이루어져 있는 2차원 데이터 입니다.
# 인공 신경망은 모든 값이 일렬로 나란히 있는 배열을 입력으로 갖습니다.
# 따라서 인공 신경망의 입력으로 2차원 이미지를 넣고 싶다면 1차원으로 모양을 변경해야 합니다.

import torch
import torch.nn as nn
from torch.optim.adam import Adam

# 학습에 사용할 장치 지정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

model.to(device) # 모델의 파라미터를 GPU 로 보냄

lr = 1e-3
optim = Adam(model.parameters(), lr=lr)

for epoch in range(20):
    for data, label in train_loader:
        optim.zero_grad()

        # 입력 데이터 모양을 모델의 입력에 맞게 변환 (일렬로 펴주는 기능)
        # mnist 이미지는 모두 28*28(784픽셀)
        # -1 의 의미는 갯수를 상관하지 않겠다는 뜻
        # 따라서 입력텐서는 (배치, (높이*너비)) (32,784)가 된다
        # -1 넣어서 위에서 정의한 배치사이즈가 들어가게 된다.
        data = torch.reshape(data, (-1,784)).to(device)
        preds = model(data)

        loss = nn.CrossEntropyLoss()(preds, label.to(device)) # 확률 분포 사용
        loss.backward() # 오차 역전파
        optim.step() # 최적화

    # loss : 실제 정답과 모델이 예측한 값 사이의 차이
    print(f"epoch{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "C:/Users/the35/Documents/Simple_AI_Model/Mush have pytorch/model/mnist.pth")


# %%

# 4. 모델 성능 평가하기
model.load_state_dict(torch.load('../Mush have pytorch/model/mnist.pth', map_location=device))

num_corr = 0 # 분류에 성공한 전체 수

with torch.no_grad(): # 기울기를 계산하지 않음
    for data, label in test_loader:
        data = torch.reshape(data, (-1, 784)).to(device)

        output = model(data.to(device))

        # max(1)[1] 가장 높은 값을 갖는 위치를 반환
        # 모든 텐서의 차원은 배치, 클래스 순서이다.
        # max(0)은 배치에서 가장 높은 값을 반환
        # max(1)은 클래스 차원에서 가장 높은 값을 반환
        preds = output.data.max(1)[1] # 모델의 예측값 계산


        # 올바르게 분류한 개수
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

    print(f"Accuracy:{num_corr/len(test_data)}") # 정확도 출력
# %%
