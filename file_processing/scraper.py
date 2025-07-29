"""
This file scrapes all the songs from a YouTube Music channel.

Main function:
    download_from_youtube_music
"""
import logging
import os
from typing import List

import yt_dlp
import pandas as pd
from time import sleep
from urllib.parse import urlparse

from classes.constants import BANNED_WORDS
from classes.logger import create_logger


def has_banned_words(title: str, banned_words: List[str]=BANNED_WORDS):
    """ [Helper function for download_track]
    Ignores songs with a banned word in the title.
    """
    for banned_word in banned_words:
        if banned_word in title.lower():
            return True, banned_word
    return False, None

def download_track(track_url, output_dir="downloads"):
    """ [Helper function for download_from_youtube_music]
    Download a single track.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'ignoreerrors': True,
        'quiet': False,
        'keepvideo': False,
        'download_archive': 'logs/downloaded_videos.txt',
        'no_overwrites': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            title = ydl.extract_info(track_url, download=False).get('title')
            has_banned_word, banned_word = has_banned_words(title)
            if has_banned_word:
                logging.info(f'Skipped {title}. Contains banned word: {banned_word}.')
                return True
            logging.info(f"Downloading {title}.")
            ydl.download([track_url])
        return True
    except Exception as e:
        logging.warning(f"Error downloading {track_url}: {e}")
        return False

def is_youtube_music_url(url):
    """ [Helper function for download_from_youtube_music]
    Check if the URL is from YouTube Music.
    """
    parsed_url = urlparse(url)
    return parsed_url.netloc in ['music.youtube.com', 'www.music.youtube.com']

def get_playlist_from_artist(artist_url):
    """ [Helper function for download_from_youtube_music]
    Extract playlist URLs from an artist page.
    """
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(artist_url, download=False)
        playlists = []
        if 'entries' in info:
            for entry in info['entries']:
                if 'url' in entry:
                    playlists.append(entry['url'])
        return playlists

def download_from_youtube_music(url, output_dir="downloads"):
    """
    Given a url to a YouTube Music channel, downloads all songs to output_dir.
    This function will ignore songs whose titles include a word in BANNED_WORDS.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not is_youtube_music_url(url):
        logging.warning("Not a YouTube Music URL")
        return
    
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    # Handle different URL types
    if '/channel/' in path or '/artist/' in path:
        logging.info(f"Processing artist: {url}")
        playlists = get_playlist_from_artist(url)
        
        for song_url in playlists:
            logging.info(f"Processing song: {song_url}")
            download_track(song_url, output_dir)

    else:
        logging.warning(f"Unsupported URL type: {url}")

# Example usage
if __name__ == "__main__":
    create_logger('logs/scrape_without_dupes.log')
    data_path = 'file_processing/DownloadViaChannel.xlsx'
    df = pd.read_excel(data_path)
    for row in df.itertuples():
        sleep(1)
        download_from_youtube_music(row.group_link, "music_downloads/" +
                                    row.company + '/' + str(row.generation) +
                                    '/' + row.group)
