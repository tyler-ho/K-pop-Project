from pytubefix import YouTube,Playlist
from typing import Optional
import os
import random
from time import sleep

'''
video: YouTube object to be downloaded
song: name of song
artist: name of artist
generation: generation of artist
output_path: destination folder
'''
def pull_audio(video: YouTube, song: str, artist: str, generation: str, output_path: Optional[str] = "audio_output") -> None:
    output_file = output_path + generation + " " + song + " - " + artist
    # prevents duplicate downloads
    if not os.path.exists(output_file):
        audio_stream = video.streams.get_audio_only()
        audio_stream.download(output_path=output_path, filename=song + " - " + artist, filename_prefix=generation + " ")

'''
link: link to playlist
'''
def pull_playlist(link: str, generation: str) -> None:
    playlist = Playlist(link)
    artist = playlist.videos[0].author[:-8]
    output_path = "audio_output" + "/" + artist
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    for video in playlist.videos:
        pull_audio(video, video.title, artist, generation=generation, output_path=output_path)
        sleep(random())

