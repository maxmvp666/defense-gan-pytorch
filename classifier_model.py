import torch
import torch.nn as nn

class Model_A(nn.Module):
    def __init__(self):
        super(Model_A, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    
class Model_B(nn.Module):
    def __init__(self):
        super(Model_B, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.softmax(x)
        return x
    
class Model_F(nn.Module):
    def __init__(self):
        super(Model_F, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        
        self.fc1 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.softmax(x)
        return x
    
class Model_C(nn.Module):
    def __init__(self):
        super(Model_C, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*13*13, 128)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
class Model_D(nn.Module):
    def __init__(self):
        super(Model_D, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 200)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
class Model_E(nn.Module):
    def __init__(self):
        super(Model_E, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x