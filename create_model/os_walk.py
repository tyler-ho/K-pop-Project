import os
from pathlib import Path
from song import Song
import pickle
import unicodedata

def createSongsFromDir(root_dir):
    full_paths = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        filenames[:] = [f for f in filenames if f != '.DS_Store' and f != 'desktop.ini']
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            full_paths.append(unicodedata.normalize('NFC', full_path))

    Songs = []

    for full_path in full_paths:
        p = Path(full_path)
        Songs.append(Song(full_path, p.parts[-4], p.parts[-3], p.parts[-2], p.parts[-1]))

    with open("create_model/Songs.pkl", "wb") as f:
        pickle.dump(Songs, f)

if __name__ == "__main__":
    root_dir = '/home/tyler/gdrive/K-pop Project 2024-5/K-pop Project/music_files'
    # root_dir = '/home/tyler/Downloads/kpop_project/music_files'
    createSongsFromDir(root_dir=root_dir)