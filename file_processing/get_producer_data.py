import logging
import os
import pickle

import dotenv
import musicbrainzngs
from musicbrainzngs import get_recording_by_id
import numpy as np
import pandas as pd

from classes.song import Song
from classes.logger import create_logger

dotenv.load_dotenv()

USER = os.getenv('MUSIC_BRAINZ_USERNAME')
PASSWORD = os.getenv('MUSIC_BRAINZ_PASSWORD')
PRODUCER_METADATA = {'arranger', 'instrument arranger', 'vocal arranger',
                     'producer', 'engineer', 'audio', 'mastering', 'sound',
                     'mix', 'recording', 'programming', 'editor', 'balance',
                     'sound_effect'}

def load_musicbrainz(user=USER, password=PASSWORD, limit_or_interval=1.0,
                     new_requests=3, fmt='xml'):
    musicbrainzngs.auth(user, password)
    musicbrainzngs.set_rate_limit(limit_or_interval=limit_or_interval,
                                  new_requests=new_requests)
    musicbrainzngs.set_useragent(app='GetKpopProducerData', 
                                 version='1.0', contact='tylerho@stanford.edu') 
    musicbrainzngs.set_hostname(new_hostname='musicbrainz.org', use_https=True)
    musicbrainzngs.set_parser()
    musicbrainzngs.set_format(fmt=fmt)


def get_producer_data(recording_id, artist, includes=['artist-rels'], 
                      release_status=[], release_type=[]): 
    if pd.isna(recording_id):
        return np.nan
    logging.info(f'Starting on recording_id {recording_id}.')
    try:
        artist_relations = get_recording_by_id(id=recording_id, includes=includes,
                                           release_status=release_status,
                                           release_type=release_type)
    except Exception as e:
        logging.warning(f'Could not find data at recording_id {recording_id} '
                        f'with exception: {e}. Returning np.nan.')
        return np.nan

    recording = artist_relations['recording']
    out = dict()
    if 'artist-relation-list' not in recording:
        logging.warn(f'No producer data for recording_id {recording_id}.')
        return out

    artist_relation_list = recording['artist-relation-list']
    for producer in artist_relation_list:
        type = producer['type']
        if type in {'vocal', 'performer'}:
            logging.warn(f'Ignoring type {type} because producer '
            f'{producer["artist"]["name"]} is in the group {artist}.')
            continue
        if type not in out:
            out[type] = set()
        out[type].add(producer['artist']['name'])

    return out

if __name__ == '__main__':
    create_logger('logs/get_producer_data.log')
    data_filepath = 'machine_data/recording_ids_list_with_nans.pkl' 
    df_filepath = 'machine_data/clip_embedding_saved_df.pkl'
    producer_list_out_path = 'machine_data/producer_list.pkl'
    df_recordingid = 'machine_data/df_clipembed_recordingid.pkl'
    df_out = 'machine_data/df_clipembed_recordingid_producermd.pkl'
    with open(data_filepath, 'rb') as f:
        data = pickle.load(f)
    with open(df_filepath, 'rb') as f:
        df = pickle.load(f)
    df['recording_id'] = data
    load_musicbrainz()
    out = []
    for row in df.itertuples():
        out.append(get_producer_data(recording_id=row.recording_id,
                                     artist=row.artist))
    with open(producer_list_out_path, 'wb') as f:
        pickle.dump(out, f)
    with open(df_recordingid, 'wb') as f:
        pickle.dump(df, f)

    df['producer_data'] = out

    with open(df_out, 'wb') as f:
        pickle.dump(df, f)
