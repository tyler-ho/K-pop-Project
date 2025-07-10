import os
import pickle
import demucs.pretrained
import demucs.separate
import librosa
import logging
import torch
import torchaudio
import numpy as np
import pandas as pd
import demucs.separate
from demucs.pretrained import get_model
from demucs.apply import apply_model
from scipy.io import wavfile
from song import Song

STEMS = ['vocals', 'no_vocals']
EXTENSIONS = [f'_{STEM}.wav' for STEM in STEMS]


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/splitting_logs.log'),
            logging.StreamHandler()
        ]
    )

def set_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_base_filename(input_file):
    input_file = os.path.abspath(input_file)
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    return base_filename
    

def separate_track(input_file, device='cpu', output_dir='stem_temp', model_name='htdemucs'):
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
    logging.info(f'Audio loaded successfully. Shape: {wav.shape}, Sample rate: {sr}')

    logging.info(f'Using device: {device}')
    model = get_model(model_name)
    model.to(device)

    if sr != model.samplerate:
        logging.info(f'Resampling from {sr}Hz to {model.samplerate}Hz')
        import torchaudio.functional as F
        wav = F.resample(wav, orig_freq=sr, new_freq=model.samplerate)

    wav = wav.unsqueeze(0)
    wav = wav.to(device)

    logging.info('Separating sources...')
    with torch.no_grad():
        sources = apply_model(model, wav)

    sources = sources.squeeze(0)
    logging.info('Saving separated sources...')
    vocals_no_vocals = [sources[3], sources[0] + sources[1] + sources [2]]

    for i, source_name in enumerate(STEMS):
            source_audio = vocals_no_vocals[i]
            output_path = os.path.join(output_dir,
                f'{base_filename}_{source_name}.wav')
            audio_numpy = source_audio.cpu().numpy()
            audio_numpy = audio_numpy.T
            logging.info(f'Saving {source_name} to {output_path}.')
            wavfile.write(output_path, model.samplerate, 
                          audio_numpy.astype(np.float32))
    logging.info(f'Separation complete! Files saved to {output_dir}')
    return

def create_clips(y, sr=44100, clip_duration=30, overlap=0):
    clip_length = clip_duration * sr
    hop_length = int(clip_length * (1 - (float(overlap) / clip_duration)))
    end_spot = int(len(y) - clip_length + 1)
    clips = []
    for start_sample in range(0, end_spot, hop_length):
        clips.append(y[start_sample:start_sample + clip_length])
    return clips

def process_librosa(audio, sr=44100):
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft.T, axis=0)

    chroma_cqt = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_cqt_mean = np.mean(chroma_cqt.T, axis=0)

    chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
    chroma_cens_mean = np.mean(chroma_cens.T, axis=0)

    chroma_vqt = librosa.feature.chroma_vqt(y=audio, sr=sr, intervals="equal")
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

    spectral_flatness = librosa.feature.spectral_flatness(y=audio)
    spectral_flatness_mean = np.mean(spectral_flatness.T, axis=0)

    poly_features = librosa.feature.poly_features(y=audio, sr=sr)
    poly_features_mean = np.mean(poly_features.T, axis=0)

    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_mean = np.mean(zcr.T, axis=0)

    tempo = librosa.feature.tempo(y=audio, sr=sr)
    tempo_mean = np.array([np.mean(tempo, axis=0)])

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
    labels.extend([f'chroma_cqt_{note}' for note in chroma_notes])
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

'''
Given librosa-loaded song stems, returns a list of features for each stem 
contained in a list.

stems: List[[stem, sr], [stem, sr], ...] 

Returns: features: List[[librosa_features], [librosa_features], ...]
'''
def process_librosa_stems(stems):
    features = []
    for i in range(len(stems)):
        features.append([])
        stem = stems[i][0]
        sr = stems[i][1]
        stem_clips = create_clips(y=stem, sr=sr)
        for clip in stem_clips:
            librosa_features = process_librosa(clip)
            features[i].append(librosa_features)
    return features


if __name__ == '__main__':    
    # unit testing
    # import time
    # input_file = "/Users/tylerho/Library/CloudStorage/GoogleDrive-tylerho@stanford.edu/.shortcut-targets-by-id/11Wd8pqP4BVeS--hw1VHHo4r5uRk9L1JP/K-pop Project 2024-5/K-pop Project/music_files/SM/3/Red Velvet/Feel My Rhythm.mp3"
    # start_time = time.perf_counter()
    # device = set_device()
    # separate_track(input_file, device=device)
    # base_filename = get_base_filename(input_file)
    # stems = [librosa.load(f'stem_temp/{base_filename}{EXTENSION}') for EXTENSION in EXTENSIONS]
    # features = process_librosa_stems(stems)
    # for EXTENSION in EXTENSIONS:
    #     os.remove(f'stem_temp/{base_filename}{EXTENSION}')
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"The process took {elapsed_time:.4f} seconds.")

    data_path = 'machine_data/pickled_songs_sherlock.pkl'
    with open(data_path, 'rb') as f:
        Songs = pickle.load(f)
    data = [vars(song) for song in Songs if type(song) == Song]
    num_none = len([song for song in Songs if song is None])
    df = pd.DataFrame(data)
    device = set_device()
    for row in df.iterrows():
        base_filename = get_base_filename(row.path)
        stems = [librosa.load(f'stem_temp/{base_filename}{EXTENSION}' for EXTENSION in EXTENSIONS)]
        features = process_librosa_stems(stems)
        row.vocal_clip_embeddings = features[0]
        row.instrumental_clip_embeddings = features[1]
        for EXTENSION in EXTENSIONS:
            os.remove(f'stem_temp/{base_filename}{EXTENSION}')
        df.to_pickle('machine_data/clip_embedding_saved_df.pkl')
