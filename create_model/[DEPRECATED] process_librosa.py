'''
This file was originally used to feed songs through a librosa feature extraction 
to take out the mfccs of each song. This file has been replaced by new_process_librosa.ipynb.
'''

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from song import Song
from typing import List
import logging
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/process_librosa.log"),
        logging.StreamHandler()
    ]
)

'''
song: the song that needs to be processed, in the Song object form
save_path: path to the output object
n_mfcc: number of Mf coefficients needed
'''
def process_sound(song: Song, save_path, n_mfcc = 13):
    if os.path.exists(save_path):
        logging.info(f"Skipping {song.path} - MFCC file already exists at {save_path}")
        return
    try:
        # Load the audio file
        y, sr = librosa.load(song.path)
        
        # Analyze the audio
        # Example: Calculate the Mel-frequency cepstral coefficients (MFCC)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Example: Calculate chroma feature
        # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Example: Calculate spectral contrast
        # contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # Print or return the desired features
        print(f"Processed {song.path}:")
        print(f"MFCC Shape: {mfccs.shape}")
        # print(f"Chroma Shape: {chroma.shape}")
        # print(f"Contrast Shape: {contrast.shape}")
        
        np.save(save_path, mfccs)
        logging.info(f"Processed and saved MFCCs for {song.path} -> {save_path}")


        # # Return extracted features
        # return {
        #     'mfccs': mfccs,
        #     # 'chroma': chroma,
        #     # 'spectral_contrast': contrast
        # }
    except Exception as e:
        logging.error(f"Error processing {song.path}: {e}")

def process_directory(Songs: List[Song], out_folder):
    # results = {}
    for song in Songs:
        save_path = os.path.join(out_folder, song.artist + " - " + song.name + ".npy")
        process_sound(song=song, save_path=save_path)
        song.features = save_path
        with open("create_model/Songs.pkl", 'wb') as f:
            pickle.dump(Songs, f)
        # results[song.path] = features
    # return results

# Example usage
if __name__ == "__main__":
    from os_walk import createSongsFromDir
    directory_path = '/home/tyler/gdrive/K-pop Project 2024-5/K-pop Project/audio_output_viachannel'
    out_folder = "/home/tyler/gdrive/K-pop Project 2024-5/K-pop Project/song_mfccs"
    with open("create_model/Songs.pkl", 'rb') as f:
        Songs = pickle.load(f)
        features_data = process_directory(Songs, out_folder)
        print(features_data)