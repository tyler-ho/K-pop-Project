import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def process_sound(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)
    
    # Analyze the audio
    # Example: Calculate the Mel-frequency cepstral coefficients (MFCC)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Example: Calculate chroma feature
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Example: Calculate spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Print or return the desired features
    print(f"Processed {file_path}:")
    print(f"MFCC Shape: {mfccs.shape}")
    print(f"Chroma Shape: {chroma.shape}")
    print(f"Contrast Shape: {contrast.shape}")
    
    # Return extracted features
    return {
        'mfccs': mfccs,
        'chroma': chroma,
        'spectral_contrast': contrast
    }

def process_directory(directory_path):
    results = {}
    for foldername, subfolders, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):  # You can include other formats supported by librosa
                file_path = os.path.join(foldername, filename)
                features = process_sound(file_path)
                results[file_path] = features
    return results

# Example usage
directory_path = 'separated_stems'
features_data = process_directory(directory_path)