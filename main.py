import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from autoencoder import AutoEncoder
from torch.utils.data import DataLoader
from torchvision import datasets
import argparse
import torchvision.utils as vutils


def get_data_loaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_autoencoder(model, train_loader, epochs, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse_loss = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    min_loss = np.inf  # 최소 손실 초기화
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mse_loss(outputs, inputs)  # MSE 손실 계산
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # 학습률 스케줄러 업데이트
        scheduler.step(avg_loss)
        if epoch % 10 == 0:
            save_results(model, train_loader, device)

        # 최소 손실 갱신 및 모델 저장
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), "./parameter/best_model.pth")
            print("Model saved with loss: {:.4f}".format(min_loss))


def save_results(model, train_loader, device):
    model.eval()

    with torch.no_grad():
        data = next(iter(train_loader))
        inputs = data[0].to(device)
        outputs = model(inputs)

        # 이미지를 0-1 사이로 클리핑
        outputs = torch.clamp(outputs, 0, 1)

        # 이미지 준비
        comparison = torch.cat([inputs[:5], outputs[:5]])

        # 이미지 저장
        vutils.save_image(comparison.cpu(), "result.png", nrow=5)


def load_pretrained_autoencoder(
    model_path="./parameter/best_model.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)

    # 모델 상태를 불러오기 전에 파일 존재 여부 확인
    if os.path.isfile(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            print("Initializing a new model.")
    else:
        print(f"No saved model found at {model_path}. Initializing a new model.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an AutoEncoder on CIFAR10")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="input batch size for training (default: 2000)",
    )
    parser.add_argument(
        "--use_pretrained", action="store_true", help="use a pre-trained model"
    )

    args = parser.parse_args()

    # 모델 초기화 및 학습
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_pretrained:
        autoencoder = load_pretrained_autoencoder().to(device)
    else:
        autoencoder = AutoEncoder().to(device)

    train_loader, _ = get_data_loaders(args.batch_size)
    train_autoencoder(autoencoder, train_loader, epochs=500, lr=0.001)
