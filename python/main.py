import os
import torch
import torch.nn as nn
import torch.optim as optim

# Defining a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # 10 inputs, 20 outputs
        self.fc2 = nn.Linear(20, 1)   # 20 inputs, 1 output
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation function
        x = self.fc2(x)
        return x

# Check if CUDA is available and configure your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a model
model = SimpleModel().to(device)

# Generating simple random data (data with 10 features)
dummy_input = torch.randn(5, 10).to(device)  # Batch size 5, feature size 10

# Check model output
output = model(dummy_input)
print("Model Output:", output)

# Save model
save_dir = "../alloc/model"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Create directory if it does not exist

model_path = os.path.join(save_dir, "simple_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Load model (for testing)
loaded_model = SimpleModel().to(device)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()  # Set to inference mode

# Predict with loaded model
with torch.no_grad():
    loaded_output = loaded_model(dummy_input)
print("Loaded Model Output:", loaded_output)
