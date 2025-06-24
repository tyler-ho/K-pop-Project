import os
from pathlib import Path
from song import Song
import pickle

def createSongsFromDir(root_dir):
    full_paths = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        filenames[:] = [f for f in filenames if f != '.DS_Store']
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            full_paths.append(full_path)

    Songs = []

    for full_path in full_paths:
        p = Path(full_path)
        Songs.append(Song(full_path, p.parts[-4], p.parts[-3], p.parts[-2], p.parts[-1]))

    with open("create_model/Songs.pkl", "wb") as f:
        pickle.dump(Songs, f)

if __name__ == "__main__":
    root_dir = '/Users/tylerho/Library/CloudStorage/GoogleDrive-tylerho@stanford.edu/.shortcut-targets-by-id/11Wd8pqP4BVeS--hw1VHHo4r5uRk9L1JP/K-pop Project 2024-5/K-pop Project/audio_output_viachannel'
    createSongsFromDir(root_dir=root_dir)