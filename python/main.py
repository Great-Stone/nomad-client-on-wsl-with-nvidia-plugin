import os
import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 신경망 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # 10개의 입력, 20개의 출력
        self.fc2 = nn.Linear(20, 1)   # 20개의 입력, 1개의 출력
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 활성화 함수
        x = self.fc2(x)
        return x

# CUDA가 사용 가능한지 확인하고, 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 생성
model = SimpleModel().to(device)

# 간단한 임의 데이터 생성 (10개 특성을 가지는 데이터)
dummy_input = torch.randn(5, 10).to(device)  # 배치 크기 5, 특성 크기 10

# 모델 출력 확인
output = model(dummy_input)
print("Model Output:", output)

# 모델 저장
save_dir = "../alloc/model"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 디렉토리가 없으면 생성

model_path = os.path.join(save_dir, "simple_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# 모델 로드 (테스트용)
loaded_model = SimpleModel().to(device)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()  # 추론 모드로 설정

# 로드한 모델로 예측
with torch.no_grad():
    loaded_output = loaded_model(dummy_input)
print("Loaded Model Output:", loaded_output)
