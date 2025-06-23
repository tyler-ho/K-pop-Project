import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from song import Song
from typing import List

def process_sound(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)
    
    # Analyze the audio
    # Example: Calculate the Mel-frequency cepstral coefficients (MFCC)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Example: Calculate chroma feature
    # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Example: Calculate spectral contrast
    # contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Print or return the desired features
    print(f"Processed {file_path}:")
    print(f"MFCC Shape: {mfccs.shape}")
    # print(f"Chroma Shape: {chroma.shape}")
    # print(f"Contrast Shape: {contrast.shape}")
    
    # Return extracted features
    return {
        'mfccs': mfccs,
        # 'chroma': chroma,
        # 'spectral_contrast': contrast
    }

def process_directory(Songs: List[Song]):
    results = {}
    for song in Songs:
        features = process_sound(song.path)
        results[song.path] = features
    return results

# Example usage
if __name__ == "__main__":
    from os_walk import createSongsFromDir

    directory_path = '/Users/tylerho/Library/CloudStorage/GoogleDrive-tylerho@stanford.edu/.shortcut-targets-by-id/11Wd8pqP4BVeS--hw1VHHo4r5uRk9L1JP/K-pop Project 2024-5/K-pop Project/audio_output_viachannel'
    Songs = createSongsFromDir(directory_path)
    features_data = process_directory(Songs)
    print(features_data)