import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import os

METADATA_FILE = 'data/metadata.csv'
DATA_DIR = 'data'
SAMPLE_RATE = 16000
NUM_SAMPLES = 48000

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

class BirdSoundDataset(Dataset):
    def __init__(self, metadata_file, data_dir, target_sample_rate, num_samples, transforms):
        self.metadata = pd.read_csv(metadata_file)
        self.data_dir = data_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.transforms = transforms

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, index):
        path = self._get_file_path(index)
        label = self._get_label(index)
        wav, sr = torchaudio.load(path)
        wav = self._resample_if_necessary(wav, sr)
        wav = self._mix_down_if_necessary(wav)
        wav = self._resize_if_necessary(wav)
        wav = self.transforms(wav)
        return wav, label

    def _resize_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        elif signal.shape[1] < self.num_samples:
            difference = self.num_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, difference))
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            signal = torchaudio.transforms.Resample(sr,self.target_sample_rate)(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0]>1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_file_path(self,index):
        filename = self.metadata.iloc[index,0]
        subfolder = self.metadata.iloc[index,1]
        return os.path.join(self.data_dir,subfolder,filename)

    def _get_label(self,index):
        return self.metadata.iloc[index,2]

if __name__ == '__main__':

    dataset = BirdSoundDataset(METADATA_FILE,DATA_DIR,SAMPLE_RATE,NUM_SAMPLES,mel_spectrogram)
    print(dataset[109][0].shape)