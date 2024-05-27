
# CNN 모델 만들기

# %%

# 1. 데이터 확인 및 전처리
import matplotlib.pyplot as plt
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor

# <데이터 셋 불러오기>
train_data = CIFAR10(root='./', train=True, download=True, transform=ToTensor)
test_data = CIFAR10(root='./', train=False, download=True, transform=ToTensor)

# 데이터셋 확인
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_data.data[i])
plt.show()

# %%
import torchvision.transforms as T
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop

# <데이터 전처리 함수>
transforms = Compose([
    T.ToPILImage(),
    RandomCrop((32,32), padding=4), # 랜덤으로 이미지 일부 제거 후 패딩
    RandomHorizontalFlip(p=0.5), # y축 기준으로 대칭
])

# 데이터 셋 불러오기
train_data = CIFAR10(root='./', train=True, download=True, transform=transforms)
test_data = CIFAR10(root='./', train=False, download=True, transform=transforms)

# 데이터셋 확인
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(transforms(train_data.data[i]))
plt.show()


# %%

from torchvision.transforms import Normalize

# <데이터 정규화>
transforms = Compose([
    T.ToPILImage(),
    RandomCrop((32,32), padding=4), # 랜덤으로 이미지 일부 제거 후 패딩
    RandomHorizontalFlip(p=0.5), # y축 기준으로 대칭
    T.ToTensor(),

    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    T.ToPILImage()
])

# 데이터 셋 불러오기
train_data = CIFAR10(root='./', train=True, download=True, transform=transforms)
test_data = CIFAR10(root='./', train=False, download=True, transform=transforms)

# 데이터셋 확인
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(transforms(train_data.data[i]))
plt.show()

# %%
