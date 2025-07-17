import logging
import pickle
import os
import acoustid
import dotenv
import numpy as np
import pandas as pd
from song import Song

dotenv.load_dotenv()

API_KEY = os.getenv('CHROMAPRINT_API_KEY')

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/audio_fingerprinting.log'),
            logging.StreamHandler()
        ]
    )

if __name__ == '__main__':
    data_path = 'machine_data/clip_embedding_saved_df.pkl'
    output_path = 'machine_data/recording_ids_list_with_nans.pkl'
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    seen_recording_ids = set()
    duplicate_recording_indices = []
    indices_to_remove = []
    recording_ids = []
    for index, row in df.iterrows():
        logging.info(f'Searching for recording_id for {row.artist} -'
                     f' {row.song_name}.')
        path = row.path
        num_recording_ids = 0 
        num_dupes = len(duplicate_recording_indices)
        for score, recording_id, title, artist in acoustid.match(API_KEY,
                                                                 path):
            # skip song if already seen before and mark it as a duplicate
            if recording_id in seen_recording_ids:
                duplicate_recording_indices.append(index)
                if num_recording_ids > 0:
                    recording_ids = recording_ids[:-1]
                recording_ids.append(np.nan)
                logging.warning(f'Duplicate song found: {row.artist} -'
                                f' {row.song_name}')
                break
            # save the first recording_id as a representative of the song
            if num_recording_ids == 0:
                recording_ids.append(recording_id)
            # save all seen recording_ids to filter out duplicate songs
            seen_recording_ids.add(recording_id)
            num_recording_ids += 1
        # if no recording ids found and there was no duplicate, then no
        # recording id could be found
        if num_recording_ids == 0 and len(duplicate_recording_indices) == num_dupes:
                logging.warning(f'No recording_id found: {row.artist} -'
                                f' {row.song_name}.')
                recording_ids.append(np.nan)
        logging.info(f'Sanity check that the number of recording_ids tracks '
                     f'with the index: {index == (len(recording_ids) - 1)}.')
    

    with open(output_path, 'wb') as f:
        pickle.dump(recording_ids, f)
