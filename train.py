# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from mlp_model import MLP

def train_model(train_loader, test_loader, epochs=5, lr=0.001, device='cpu'):
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  #优化器

    for epoch in range(epochs):
        model.train()
        total_loss = 0  #用来累计当前轮的损失
        for batch in train_loader:
            x, y = batch
            x = x.view(x.size(0), -1).to(device)  #数据预处理
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model