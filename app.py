import h5py, math, numpy as np
from IPython.display import Audio
from pydub import AudioSegment
import torchaudio
import torch
import torch.nn as nn
from StutterNet import SEP28KDataset, StutterNet, sigmoid
import matplotlib.pyplot as plt
import gradio as gr

def clip(audio, step=500, clip_len=3000):
  # clip to 3 seconds
  clips = []
  start = 0
  while start + clip_len < int(len(audio)):
    clips.append(audio[start:start+clip_len])
    start += step
  return clips

def align(audio, mean, std):
  # normalize
  audio = (audio - audio.mean()) / audio.std()
  # align with training data
  audio = audio * std + mean
  return audio

def stutterPrediction(SPEECH_FILE):
  # read the hdf5 dataset
  f = h5py.File("/content/drive/MyDrive/SEP28K.hdf5", 'r')

  # create train, test, validation splits
  trainX, trainY, testX, testY, validX, validY = f['trainX'], f['trainY'], f['testX'], f['testY'], f['validX'], f['validY']

  # Calculate sample distribution
  mean, var, std = 0, 0, 0
  for sample in trainX:
    mean += sample.mean()
    var += sample.var() + sample.mean()**2

  mean /= trainX.shape[0]
  var = var/trainX.shape[0] - mean**2
  std = math.sqrt(var)

  audio = Audio(SPEECH_FILE)
  display(audio)

  # clip
  audio = AudioSegment.from_wav(SPEECH_FILE)
  audio = audio.set_frame_rate(16000)
  audio = audio.set_channels(1)
  step, clip_len = 500, 3000
  audio = clip(audio, step, clip_len)

  for i in range(len(audio)):
    audio[i] = np.array(audio[i].get_array_of_samples())
  audio = np.array(audio)
  audio = audio.astype(np.float32)

  # align by mean & std
  for i in range(audio.shape[0]):
    audio[i] = align(audio[i], mean, std)

  # get device
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device

  # prepare data and predict
  # create required transforms
  spec = torchaudio.transforms.MelSpectrogram(n_mels=80, sample_rate=16000,
                                                n_fft=512, f_max=8000, f_min=0,
                                                power=0.5, hop_length=152, win_length=480)
  db = torchaudio.transforms.AmplitudeToDB()
  transforms = torch.jit.script(nn.Sequential(spec, db))

  # create datasets
  ds = SEP28KDataset(audio, np.ones((audio.shape[0],12)), transform=transforms)
  dataloader = torch.utils.data.DataLoader(ds, batch_size=audio.shape[0], shuffle=False, num_workers=2, pin_memory=True)

  # ensemble learning
  state = torch.load("/content/drive/MyDrive/StutterNet2.pth", map_location=device)
  net1 = StutterNet(80, dropout=0.2, scale=2).to(device)
  net1.load_state_dict(state['state_dict'])
  net1.eval()

  state = torch.load("/content/drive/MyDrive/StutterNet.pth", map_location=device)
  net2 = StutterNet(80, dropout=0.2).to(device)
  net2.load_state_dict(state['state_dict'])
  net2.eval()

  # prediction placeholders
  preds = np.zeros((len(ds), 12))

  for data in iter(dataloader):
    # get features and labels
    inputs, labels = data[0].to(device), data[1].detach().cpu().numpy()

    # get predictions
    preds = (net1(inputs).detach().cpu().numpy() + net2(inputs).detach().cpu().numpy()) / 2

  preds = sigmoid(preds)

  names = np.loadtxt('classes.txt', dtype=str)

  dpreds = np.zeros(preds.shape)

  dpreds[0] = preds[0]
  for i in range(1, preds.shape[0]):
    dpreds[i] = preds[i] - preds[i-1]

  # ad-hoc
  first_threshold = 0.3
  second_threshold = 0.2
  cur_stat = None
  pred = []
  for i in range(preds.shape[0]):
    ac = np.argmax(dpreds[i])
    pr = np.argmax(preds[i])
    if cur_stat != None and dpreds[i,cur_stat] < -second_threshold:
      cur_stat = None
    if dpreds[i,ac] > second_threshold:
      cur_stat = ac
    if cur_stat == None and preds[i,pr] > first_threshold:
      cur_stat = pr
    if cur_stat == None:
      pred.append('None')
    else:
      pred.append(names[cur_stat])

  # plot
  fig = plt.figure()
  plt.plot(np.arange(preds.shape[0])*step/1000, pred)
  plt.title("Classification result")
  plt.xlabel("Seconds (time-axis)")
  plt.ylabel("Class")
  plt.tight_layout()

  return fig

demo = gr.Interface(
    fn = stutterPrediction,
    inputs = gr.Audio(sources="microphone", type="filepath"),
    outputs = [gr.Plot(label="Predicted map")]
)

demo.launch(share=True)
