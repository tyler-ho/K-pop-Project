import yt_dlp
import os
import pickle
import logging
import pandas as pd
from urllib.parse import urlparse
from song import Song, debut_to_generation

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler('logs/create_songs.log'),
            logging.StreamHandler()
            ]
        )

def is_youtube_music_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc in ['music.youtube.com', 'www.music.youtube.com']

def get_video_info(url):
    ydl_opts = {
            'no_warnings': True
            }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title')
        release_date = info.get('release_date')
        return {
                'release_date': release_date,
                'title': title
                }

def get_playlists_from_artist(artist_url):
    ydl_opts = {
            'extract_flat': True,
            'skip_download': True,
            'quiet': True
            }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(artist_url, download=False)
        playlists = []
        if 'entries' not in info:
            return playlists

        for entry in info['entries']:
            if 'url' in entry:
                logging.info(f'URL was found at {entry["url"]}.')
                playlists.append(entry['url'])
        return playlists

def create_song(company: str, debut_year: int, generation: int, artist: str, title: str, release_date: str):
    file_path = f'/scratch/users/tylerho/kpop_project_data/music_downloads/{company}/{generation}/{artist}/{title}.mp3'
    if os.path.exists(file_path):
        logging.info(f'Creating the Song object for {artist} - {title}.')
        return Song(path=file_path, company=company, debut_year=debut_year,
                artist = artist, song_name=title, release_date=release_date)

def create_songs_from_channel(url, company, debut_year, generation,
        artist):
    if not is_youtube_music_url(url):
        logging.warning("Not a YouTube Music URL")
        return
    parsed_url = urlparse(url)
    path = parsed_url.path
    songs = []

    if '/channel/' in path or '/artist/' in path:
        playlists = get_playlists_from_artist(url)
        for song_url in playlists:
            logging.info(f'Processing song {url}.')
            video_info = get_video_info(song_url)
            song = create_song(company=company, debut_year=debut_year,
                    generation=generation, artist=artist,
                    title=video_info['title'],
                    release_date=video_info['release_date'])
            songs.append(song)
    return songs

if __name__ == '__main__':
    data_path = 'file_processing/DownloadViaChannel.xlsx'
    df = pd.read_excel(data_path)
    songs = []
    def debut_to_generation(debut_year: int) -> int:
        if debut_year <= 2004:
            return 1
        elif debut_year <= 2011:
            return 2
        elif debut_year <=  2017:
            return 3
        elif debut_year <= 2022:
            return 4
        else:
            return 5

    for row in df.itertuples():
        logging.info(f'Beginning on artist: {row.group}.')
        songs.extend(create_songs_from_channel(row.group_link,
            row.company, row.debut_year,
            debut_to_generation(row.debut_year), row.group))
    with open('machine_data/pickled_songs_sherlock.pkl', 'wb') as f:
        pickle.dump(songs, f)
