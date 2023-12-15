import torch
from torch import nn
import torchaudio
from torch.utils.data import DataLoader
from dataset import BirdSoundDataset
from cnn import CNNNetwork
from dataset import METADATA_FILE, DATA_DIR, SAMPLE_RATE, NUM_SAMPLES, mel_spectrogram

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        target = target.unsqueeze(1)
        target = target.float()
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'using {device}')

    dataset = BirdSoundDataset(METADATA_FILE,DATA_DIR,SAMPLE_RATE,NUM_SAMPLES,mel_spectrogram)

    train_data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    cnn = CNNNetwork().to(device)
    print(cnn)

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(params=cnn.parameters(),lr=LEARNING_RATE)
    train(cnn, train_data_loader, loss_function, optimizer, device, EPOCHS)

    torch.save(cnn.state_dict(), 'weights.pth')