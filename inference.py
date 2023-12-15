import torch
import torchaudio
import os
import pandas as pd
import random
from dataset import BirdSoundDataset
from cnn import CNNNetwork
from dataset import METADATA_FILE, DATA_DIR, SAMPLE_RATE, NUM_SAMPLES, mel_spectrogram

class_mapping = ["Capuchin","Not_Capuchin"]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        if predictions>0.5:
          predicted_index = 1
        else:
          predicted_index = 0
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

def make_prediction_on_audio(audio_file):
  signal, sr = torchaudio.load(audio_file)
  signal = torchaudio.transforms.Resample(sr,SAMPLE_RATE)(signal)
  signal = torch.mean(signal, dim=0, keepdim=True)
  num_splits = int(signal.shape[1]/NUM_SAMPLES)
  prediction_list = []
  for i in range(num_splits):
    sample = signal[:,i*NUM_SAMPLES:(i+1)*NUM_SAMPLES]
    input = mel_spectrogram(sample)
    with torch.no_grad():
        input.unsqueeze_(0)
        prediction = cnn(input)
        if prediction>0.5:
          prediction_list.append(1)
        else:
          prediction_list.append(0)
  return prediction_list

def group_clusters(arr):
  new_arr = []
  f = arr[0]
  for i in arr:
    if i != f:
      new_arr.append(f)
      f = i
  if new_arr == []:
    return 0
  else:
    return sum(new_arr)

if __name__ == '__main__':
  cnn = CNNNetwork()
  state_dict = torch.load("weights.pth")
  cnn.load_state_dict(state_dict)

  # load dataset
  dataset = BirdSoundDataset(METADATA_FILE,DATA_DIR,SAMPLE_RATE,NUM_SAMPLES,mel_spectrogram)
  
  # get a sample from dataset for inference
  random_int = random.randint(0,len(dataset))
  input, target = dataset[random_int][0], dataset[random_int][1]
  input.unsqueeze_(0)
  predicted, expected = predict(cnn, input, target, class_mapping)
  print(f"For a random dataset point, Predicted: '{predicted}', expected: '{expected}'")

  # Making inference on forest data
  results = []
  for file in os.listdir(r"data/Forest Recordings"): # please try renaming folder to remove spaces, if any errors occurs
    array = make_prediction_on_audio(os.path.join(r"data/Forest Recordings",file))
    
    # grouping consecutive calls as one call
    results.append({'filename':file,'calls':group_clusters(array)})

  final_df = pd.DataFrame(results)
  final_df.to_csv('results.csv',index=False)