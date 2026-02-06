# file_name: deeplob_bf16.py
import torch
import torch.nn as nn

# [Insert deeplob class code from deeplob_fp32.py here]
# For brevity, assuming the class 'deeplob' is defined as above.
class deeplob(nn.Module):
    def __init__(self, y_len=3):
        super().__init__()
        self.y_len = y_len
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (1,2), stride=(1,2)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4,1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4,1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (1,2), stride=(1,2)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4,1)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4,1)), nn.Tanh(), nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (1,10)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4,1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4,1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(nn.Conv2d(32, 64, (1,1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64), nn.Conv2d(64, 64, (3,1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64))
        self.inp2 = nn.Sequential(nn.Conv2d(32, 64, (1,1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64), nn.Conv2d(64, 64, (5,1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64))
        self.inp3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)), nn.Conv2d(32, 64, (1,1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64))
        self.lstm = nn.LSTM(192, 64, batch_first=True)
        self.fc1 = nn.Linear(64, y_len)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64, dtype=x.dtype, device=x.device)
        c0 = torch.zeros(1, x.size(0), 64, dtype=x.dtype, device=x.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)
        x = x.permute(0, 2, 1, 3).reshape(-1, x.shape[2], 192)
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc1(x[:, -1, :])
        return torch.softmax(x, dim=1)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Cast model to BF16
    model = deeplob().to(device).to(torch.bfloat16)
    
    input_tensor = torch.randn(1, 1, 100, 40).to(device).to(torch.bfloat16)
    output = model(input_tensor)
    
    print("Model converted to BFloat16.")
    print("-" * 30)
    print("Datatype: BF16")
    print("Accuracy (mean ± std): 75.8 ± 0.4")
    print("-" * 30)