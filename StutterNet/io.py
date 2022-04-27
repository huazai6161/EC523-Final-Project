import torch
import numpy as np
import pandas as pd
import librosa

class SEP28KDataset(torch.utils.data.Dataset):
    """SEP-28k Dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        annotations = pd.read_csv(csv_file)
        self.info = annotations.iloc[:, :3].copy()
        self.labels = annotations[["PoorAudioQuality", "Unsure",
          "DifficultToUnderstand", "NaturalPause","Music","NoSpeech",
          "Prolongation","Block","SoundRep","WordRep","Interjection",
          "NoStutteredWords"]].copy()
        # [0.11, 0.25, 0.40, 0.62, 0.57, 0.39]
        # self.spec = audio.transforms.MelSpectrogram(n_mels=40, sample_rate=16000,
        #                                        n_fft=512, pad=1, f_max=8000, f_min=0,
        #                                        power=0.5, hop_length=160, win_length=480)
        self.spec = audio.transforms.MelSpectrogram(n_mels=80, sample_rate=16000,
                                               n_fft=512, f_max=8000, f_min=0,
                                               power=0.5, hop_length=152, win_length=480)
        self.db = audio.transforms.AmplitudeToDB()

        self.freq_mask = audio.transforms.FrequencyMasking(freq_mask_param=1)
        # self.time_mask = audio.transforms.TimeMasking(freq_mask_param=20)

        self.rng = np.random.default_rng(42)
        # self.rng_2 = np.random.default_rng(68)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get path attributes and clip interval
        row = self.info.iloc[idx, :]
        
        # get clip path
        clip_path = f"{self.root_dir}/{row[0]}/{row[1]}/{row[0]}_{row[1]}_{row[2]}.wav"

        # load sliced clip
        # _, wav = wavfile.read(clip_path)
        wav, _ = librosa.load(clip_path, 16000)
        wav = self.pad_trunc(wav, 3000, 16000).astype('float32')

        wav = torch.tensor(wav)
        wav = self.spec(wav)
        wav = self.db(wav)

        if (self.rng.choice(2,p=[0.2,0.8])):
          wav = self.freq_mask(wav)

        # if (self.rng_2.choice(2,p=[0.2,0.8])):
        #   wav = self.time_mask(wav)

        # wav = torch.unsqueeze(wav, 0)

        # get labels
        labels = self.labels.iloc[idx, :].values
        labels = (labels >= 2).astype('float32')

        # ndl = np.array([ndl])
        # ndl = ndl.astype('float').reshape(-1, 1)

        # sl = np.array([sl])
        # sl = sl.astype('float').reshape(-1, 1)

        # labels = np.array([labels])
        # labels = labels.astype('float').reshape(-1, 1)
        # sample = {'clip': wav, 'stuttering': sl, 'non_stuttering': ndl}

        # if self.transform:
        #     sample = self.transform(sample)

        return torch.tensor(wav), torch.tensor(labels)
        
    @staticmethod
    def pad_trunc(sig, max_ms, sr):
      sig_len = sig.shape[0]
      max_len = sr//1000 * max_ms

      if (sig_len > max_len):
        # Truncate the signal to the given length
        sig = sig[:,:max_len]

      elif (sig_len < max_len):
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = np.random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with 0s
        pad_begin = np.zeros((pad_begin_len))
        pad_end = np.zeros((pad_end_len))

        sig = np.concatenate((pad_begin, sig, pad_end), 0)
        
      return sig
