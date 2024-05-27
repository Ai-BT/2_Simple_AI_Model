# 입력된 이미지를 10가지 항목으로 분류하는 CNN 모델

# 알고리즘 : VGG
# 데이터셋 : CIFAT-10
# 10가지 사물과 동물로 이루어진 간단한 데이터셋

# nn.Sequential
# 층을 쌓기만 하는 간단한 구조에서 사용하기 편리
# 하지만 은닉층에서 순전파 도중의 결과를 저장하거나 데이터 흐름을 제어하는 등의 커스텀마이징은 불가

# nn.Module
# 자신이 원하는대로 신경망 동작을 정의 할 수 있음
# 복잡한 신경망은 Module을 사용하는 것이 좋다

# CNN 기본 블록
# 입력 -> 합성곱 3X3 -> Relu -> 합성곱 3X3 -> Relu -> 맥스풀링 -> 출력

# %%

# VGG 기본 블록 정의
import torch
import torch.nn as nn

# 1. 기본 블록 정의
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim): # 기본 블록 정의
        # 기본 블록을 구성하는 층 정의
        super(BasicBlock, self).__init__() # nn.Module 클래스 요소 상속

        # 합성곱층 정의
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # stride는 커널의 이동 거리
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # 기본 순전파 정의
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        return x

# %%

# 2. CNN 모델 - 블록도 참고 (p.133)
class CNN(nn.Module):
    def __init__(self, num_classes): # num_calsses 클래스 개수
        super(CNN, self).__init__()

        # 합성곱 기본 블록 정의
        self.block1 = BasicBlock(in_channels=3, out_channels=32, hidden_dim=16)
        self.block2 = BasicBlock(in_channels=32, out_channels=128, hidden_dim=64)
        self.block3 = BasicBlock(in_channels=128, out_channels=256, hidden_dim=128)
        
        # 분류기 정의
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)

        # 분류기 활성화 함수
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

# %%

# 3. 모델 정의 및 학습하기
# 모델 정의
# 데이터 호출
# 손실 계산
# 오차 역전파 및 최적화 (가중치 업데이트)
# 원하는 만큼 반복

# <데이터 확인 및 전처리>
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torchvision.datasets.cifar import CIFAR10

import torchvision.transforms as T
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize

transforms = Compose([
    RandomCrop((32,32), padding=4), # 랜덤으로 이미지 일부 제거 후 패딩
    RandomHorizontalFlip(p=0.5), # y축 기준으로 대칭
    T.ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
])

# 데이터 셋 불러오기
train_data = CIFAR10(root='./', train=True, download=True, transform=transforms)
test_data = CIFAR10(root='./', train=False, download=True, transform=transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN(num_classes=10)

model.to(device)

# %%

# 4. 모델 학습하기

lr = 1e-3
optim = Adam(model.parameters(), lr=lr)

for epoch in range(10):
    for data, label in train_loader:
        optim.zero_grad()

        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device)) # 확률 분포 사용
        loss.backward() # 오차 역전파
        optim.step() # 최적화

    # loss : 실제 정답과 모델이 예측한 값 사이의 차이
    if epoch == 0 or epoch%10 == 9:
        print(f"epoch{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "C:/Users/the35/Documents/Simple_AI_Model/Mush have pytorch/model/cifar.pth")


# %%

# 5. 모델 성능 평가하기
model.load_state_dict(torch.load('../Mush have pytorch/model/cifar.pth', map_location=device))

num_corr = 0

with torch.no_grad(): 
    for data, label in test_loader:
        
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

    print(f"Accuracy:{num_corr/len(test_data)}") # 정확도 출력
# %%
