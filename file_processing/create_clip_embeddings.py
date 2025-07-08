import pickle
import librosa
import logging
import numpy as np
import pandas as pd
from demucs.pretrained import get_model
from demucs.apply import apply_model
from scipy.io import wavfile
from song import Song

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/splitting_logs.log'),
            logging.StreamHandler()
        ]
    )


def separate_track(input_file, output_dir='stem_temp',
        model_name='htmdemucs_2s'):
    input_file = os.path.abspath(input_file)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'Audio file not found: {input_file}.')
    logging.info(f'Processing file: {input_file}')

    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    os.makedirs(output_dir, exist_ok=True)

    logging.info('Loading audio with librosa...')
    audio_data, sr =  librosa.load(input_file, sr=None, mono=False)

    if len(audio_data.shape) == 1 or audio_data.shape[0] == 1:
        audio_data = np.stack([audio_data, audio_data])
        if len(audio_data.shape) == 3:
            audio_data = audio_data.squeeze(1)
    elif audio_data.shape[0] > 2:
        audio_data = audio_data[:2]

    wav = torch.tensor(audio_data, dtype=torch.float32)
    logging.info(f'Audio loaded successfully. Shape: {wave.shape}, Sample rate:
    {sr}')

    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    logging.info(f'Using device: {device}')
    model = get_model(model_name)
    model.to(device)

#   if sr != model.samplerate:
#       logging.infor(f'Resampling from {sr}Hz to {model.samplerate}Hz')
#       import torchaudio.functional as F
#       wav = F.resample(wav, orig_freq=sr, new_freq=model.samplerate)

    wav = wav.unsqueeze(0)
    wav = wav.to(device)

    logging.info('Separating sources...')
    with torch.no_grad():
        sources.apply_model(model, wav)

    sources = sources.squeeze(0)
#   logging.info('Saving separated sources...')
#   for i, source_name in enumerate(model.sources):
#           source_audio = sources[i]
#           output_path = os.path.join(output_dir,
#               f'{base_filename}_{source_name}.wav')
#           audio_numpy = source_audio.cpu().numpy()
#           audio_numpy = audio_numpy.T
#           logging.info(f'Saving {source_name} to {output_path}.')
#           wavfile.write(output_path, model.samplerate,
#               audio_numpy.astype(np.float32)
    logging.info(f'Separation complete! Files saved to {output_dir}')
    return model.sources

def create_clips(y, sr, clip_duration=30, overlap=0):
    clip_length = clip_duration * sr
    hop_length = int(clip_length * (1 - (float(overlap) / clip_duration)))
    end_spot = int(len(y) - clip_length + 1)
    clips = []
    for start_sample in range(0, end_spot, hop_length):
        clips.append(y[start_sample:start_sample + clip_length])
    return clips

def process_librosa(audio, sr):
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft.T, axis=0)

    chroma_cqt = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_cqt_mean = np.mean(chroma_cqt.T, axis=0)

    chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
    chroma_cens_mean = np.mean(chroma_cens.T, axis=0)

    chroma_vqt = librosa.feature.chroma_vqt(y=audio, sr=sr)
    chroma_vqt = np.mean(chroma_vqt.T, axis=0)

    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    melspectrogram_mean = np.mean(melspectrogram.T, axis=0)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    rmse = librosa.feature.rms(y=audio)
    rmse_mean = np.mean(rmse.T, axis=0)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid.T, axis=0)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth.T, axis=0)

    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

    spectral_flatness = librosa.feature.spectral_flatness(y=audio, sr=sr)
    spectral_flatness_mean = np.mean(spectral_flatness.T, axis=0)

    poly_features = librosa.feature.poly_features(y=audio, sr=sr)
    poly_features_mean = np.mean(poly_features.T, axis=0)

    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_mean = np.mean(zcr.T, axis=0)

    tempo = librosa.feature.tempo(y=audio, sr=sr)
    tempo_mean = np.mean(tempo.T, axis=0)

    tempogram = librosa.feature.tempogram(y=audio, sr=sr)
    tempogram_mean = np.mean(tempogram.T, axis=0)

    fourier_tempogram = librosa.feature.fourier_tempogram(y=audio, sr=sr)
    fourier_tempogram_mean = np.mean(fourier_tempogram.T, axis=0)

    tempogram_ratio = librosa.feature.tempogram_ratio(y=audio, sr=sr)
    tempogram_ratio_mean = np.mean(tempogram_ratio.T, axis=0)

    features = np.concatenate((
        chroma_stft_mean, chroma_cqt_mean, chroma_cens_mean, chroma_vqt,
        melspectrogram_mean, mfccs_mean, rmse_mean, spectral_centroid_mean,
        spectral_bandwidth_mean, spectral_contrast_mean,
        spectral_flatness_mean, poly_features_mean, tonnetz_mean, zcr_mean,
        tempo_mean, tempogram_mean, fourier_tempogram_mean,
        tempogram_ratio_mean
        ))
    return features

def get_feature_labels():
    labels = []
    chroma_notes =  ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A',
            'A#', 'B']
    labels.extend([f'chroma_stft_{note}' for note in chroma_notes])
    labels.extend([f'chroma_cqt_[note}' for note in chroma_notes])
    labels.extend([f'chroma_cens_{note}' for note in chroma_notes])
    labels.extend([f'chroma_vqt_{note}' for note in chroma_notes])

    labels.extend([f'mel_band{i}' for i in range(128)])
    labels.extend([f'mfcc_{i}' for i in range(1, 14)])
    labels.append('rms_energy')
    labels.append('spectral_centroid')
    labels.append('spectral_bandwidth')
    labels.append([f'spectral_contrast_band_{i}' for i in range(1,8)])
    labels.append('spectral_flatness')
    labels.extend(['poly_feature_0', 'poly_feature_1'])
    labels.extend([f'tonnetz_{i}' for i in range(1,7)])
    labels.append('zero_crossing_rate')

    labels.append('tempo')
    labels.extend([f'tempogram_{i}' for i in range(384)])
    labels.extend([f'fourier_tempogram_{i}' for i in range(384)])
    labels.extend([f'tempogram_ratio_{i}' for i in range(8)])

    return labels

if __name__ == '__main__':
    ...
