# 사인 함수를 3차 다항식의 계수를 이용해서 예측하는 문제

# 알고리즘: MLP (Multi-Layer Perceptron)
# 문제유형: 회귀
# 평가지표: 평균 제곱 오차
# 주요 패키지: torch, torch.nn

# 파이토치 학습 과정
# 1. 모델 정의
# 2. 모델 순전파
# 3. 오차 계산
# 4. 오차 역전파(가중치 업데이트)
# 5. 원하는 만큼 반복


# %%

# 필요한 라이브러리 불러오기
import math
import torch
import matplotlib.pyplot as plt

# 1
# -pi 부터 pi 사이에서 점을 1,000 개 추출
# linespace 시작점 a 부터 종료점 b 까지 c 개를 반환
# 즉, -파이 부터 +파이 까지 1,000개 추출
x = torch.linspace(-math.pi, math.pi, 1000)

# 2
# 실제 사인곡선에서 추출한 값으로 y 만들기
y = torch.sin(x)

# 3
# 예측 사인곡선에서 사용할 임의의 가충치(계수)를 뽑아 y 만들기
a = torch.randn(()) # 정규분포를 따르는 랜덤한 값을 반환
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

# 사인 함수를 근사할 3차 다항식 정의
y_random = a * x**3 + b * x**2 + c * x + d

# 학습률 정의
learning_rate = 1e-6

# 학습 진행
for epoch in range(2000):
    y_pred = a * x**3 + b * x**2 + c * x + d

    loss = (y_pred - y).pow(2).sum().item() # 손실 정의
    if epoch % 100 == 0:
        print(f"epoch{epoch + 1} loss:{loss}")
    
    grad_y_pred = 2.0 * (y_pred - y) # 기울기의 미분값
    grad_a = (grad_y_pred * x ** 3).sum()
    grad_b = (grad_y_pred * x ** 2).sum()
    grad_c = (grad_y_pred * x).sum()
    grad_d = grad_y_pred.sum()

    a -= learning_rate * grad_a # 가중치 업데이트
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c 
    d -= learning_rate * grad_d

# 4
# 실제 사인곡선 그리기
plt.subplot(3, 1, 1) # (행의 개수, 열의 개수, 위치)
plt.title("y true")
plt.plot(x,y)

# 5
# 예측 사인곡선 그리기
plt.subplot(3, 1, 2)
plt.title("y_pred")
plt.plot(x,y_pred)

# 6
# 랜덤 사인곡선 그리기
plt.subplot(3, 1, 3)
plt.title("y_random")
plt.plot(x,y_random)


# 7
# 출력
plt.show()

# %%
