# main.py
import torch
from utils import load_data
from train import train_model

def evaluate(model, test_loader, device='cpu'):#模型评估函数
    model.eval() #设置模型为评估模式
    correct = 0 #正确预测的数量
    total = 0 #样本总数
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.size(0), -1).to(device)#将图像拉成向量
            y = y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = load_data()
    model = train_model(train_loader, test_loader, epochs=5, device=device)
    evaluate(model, test_loader, device)
