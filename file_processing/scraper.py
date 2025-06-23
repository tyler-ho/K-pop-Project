import yt_dlp
import os
import re
from urllib.parse import urlparse, parse_qs

def is_youtube_music_url(url):
    """Check if the URL is from YouTube Music"""
    parsed_url = urlparse(url)
    return parsed_url.netloc in ['music.youtube.com', 'www.music.youtube.com']

def get_playlist_from_artist(artist_url):
    """Extract playlist URLs from an artist page"""
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(artist_url, download=False)
        playlists = []
        
        if 'entries' in info:
            for entry in info['entries']:
                if 'url' in entry:
                    playlists.append(entry['url'])
        
        return playlists

def download_track(track_url, output_dir="downloads"):
    """Download a single track"""
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
        'download_archive': 'downloaded_videos.txt',
        'no_overwrites': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([track_url])
        return True
    except Exception as e:
        print(f"Error downloading {track_url}: {e}")
        return False

def download_from_youtube_music(url, output_dir="downloads"):
    """Main function to download from YouTube Music URLs"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not is_youtube_music_url(url):
        print("Not a YouTube Music URL")
        return
    
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    # Handle different URL types
    if '/channel/' in path or '/artist/' in path:
        print(f"Processing artist: {url}")
        playlists = get_playlist_from_artist(url)
        
        for playlist_url in playlists:
            print(f"Processing playlist: {playlist_url}")
            download_track(playlist_url, output_dir)
    
    elif '/playlist' in path:
        print(f"Processing playlist: {url}")
        download_track(url, output_dir)
    
    elif '/watch' in path:
        print(f"Processing single track: {url}")
        download_track(url, output_dir)
    
    else:
        print(f"Unsupported URL type: {url}")

# Example usage
# if __name__ == "__main__":
#     artist_url = "https://music.youtube.com/channel/UCkbbMCA40i18i7UdjayMPAg"
#     download_from_youtube_music(artist_url, "audio_output_viachannel/Blackpink")



# pull_playlist("https://music.youtube.com/playlist?list=OLAK5uy_km8YKnQCFd510fbMib2ZhDKYWKWvz_qls", "5", "Xdinary Heroes")
# pull_all_songs_from_channel("https://music.youtube.com/channel/UCkbbMCA40i18i7UdjayMPAg", "YG", "Blackpink", "3")