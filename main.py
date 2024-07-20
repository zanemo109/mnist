import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
import torch.nn as nn
from matplotlib import pyplot as plt

from model import MLP

hidden_1_size = 50
hidden_2_size = 20
EPOCHS = 1000
lr = 0.1

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')


data = train_df.to_numpy()
labels = data[:, 0]
pixels = data[:, 1:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

idx = np.random.permutation(len(pixels))
shuffled_pixels = pixels[idx]
shuffled_labels = labels[idx]

test_train_split = int(0.8 * len(pixels))

train_pixels, val_pixels = shuffled_pixels[:test_train_split], shuffled_pixels[test_train_split:]
train_labels, val_labels = shuffled_labels[:test_train_split], shuffled_labels[test_train_split:]

train_pixels_tensor, train_labels_tensor = torch.Tensor(train_pixels), torch.LongTensor(train_labels)
val_pixels_tensor, val_labels_tensor = torch.Tensor(val_pixels), torch.LongTensor(val_labels)

train_pixels_tensor, train_labels_tensor = train_pixels_tensor.to(device), train_labels_tensor.to(device)
val_pixels_tensor, val_labels_tensor = val_pixels_tensor.to(device), val_labels_tensor.to(device)

input_size = 784
hidden_1_size = 100
hidden_2_size = 50
output_size = 10

model = MLP(input_size, hidden_1_size, hidden_2_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

EPOCHS = 500

def get_accuracy(predictions, actual):
    pred_labels = predictions.argmax(dim=1)
    return (pred_labels == actual).float().mean().item()

for iteration in range(EPOCHS):
    optimizer.zero_grad()
    output = model(train_pixels_tensor)
    train_loss = criterion(output, train_labels_tensor)

    train_accuracy = get_accuracy(output, train_labels_tensor)

    train_loss.backward()
    optimizer.step()

    if iteration % 10 == 0:
        print(f'ITERATION {iteration} - Training accuracy: {train_accuracy}, Train loss: {train_loss.item()}')
    
    if iteration % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_output = model(val_pixels_tensor)
            val_loss = criterion(val_output, val_labels_tensor)
            val_accuracy = get_accuracy(val_output, val_labels_tensor)
        print(f'ITERATION {iteration} - Validation accuracy: {val_accuracy}, Validation loss: {val_loss.item()}')

print(f'''FINAL TRAIN ACCURACY: {train_accuracy}
FINAL TRAIN LOSS: {train_loss.item()}
FINAL VALIDATION ACCURACY: {val_accuracy}
FINAL VALIDATION LOSS: {val_loss.item()}
''')

def get_prediction_digit(tensor):
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1)
    return prediction


def show_random_predictions(model, val_pixels_tensor, val_labels_tensor, num_predictions):
    model.eval()
    with torch.no_grad():
        indices = np.random.choice(len(val_pixels_tensor), num_predictions, replace=False)
        samples = val_pixels_tensor[indices]
        actual_labels = val_labels_tensor[indices]
        predictions = get_prediction_digit(samples)

        for i, index in enumerate(indices):
            plt.subplot(2, 5, i+1)
            plt.imshow(samples[i].cpu().numpy().reshape(28, 28), cmap='gray')
            plt.title(f'Pred: {predictions[i].item()}\nActual: {actual_labels[i].item()}')
            plt.axis('off')
        plt.show()

show_random_predictions(model, val_pixels_tensor, val_labels_tensor, 10)













