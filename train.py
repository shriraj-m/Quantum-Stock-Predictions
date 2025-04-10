import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import HybridQNN
from data_loader import load_stock_data


def train_model(ticker="AAPL", sequence_length=7, epochs=25, lr=0.001, hidden_dim=32):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Training on {device}")
    
    # load data and move to device
    X_train, X_test, y_train, y_test, scaler = load_stock_data(ticker, sequence_length)
    # X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)
    # y_train, y_test = y_train.to(device), y_test.to(device)

    # init model
    model = HybridQNN(input_dim=sequence_length, hidden_dim=hidden_dim)
    # move to device
    # model.to(device)

    # define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    train_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")


    # evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        true_values = y_test.numpy()

    # plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # inverse scale predictions for plotting if needed
    pred_prices = scaler.inverse_transform(predictions)
    true_prices = scaler.inverse_transform(true_values)

    # plot predictions and true values
    plt.figure(figsize=(10, 5))
    plt.plot(pred_prices, label="Predictions")
    plt.plot(true_prices, label="True Values")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model


if __name__ == "__main__":
    train_model(ticker="NVDA", sequence_length=7, epochs=25)
