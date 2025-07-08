def debut_to_generation(debut_year: int) -> int:
    if debut_year <= 2004: # beginning of K-pop-2004
        return 1
    elif debut_year <= 2011: # 2005-2011
        return 2
    elif debut_year <= 2017: # 2012-2017
        return 3
    elif debut_year <= 2022: # 2018-2022
        return 4
    else: # 2023-present
        return 5

class Song:
    def __init__(self, path: str, company: str, debut_year: int, artist: str, song_name: str, release_date: str):
        self.path = path
        self.company = company
        self.debut_year = debut_year
        self.generation = debut_to_generation(self.debut_year)
        self.company_generation = f"({company}, {str(self.generation)})"
        self.artist = artist
        self.song_name = song_name
        self.release_date = release_date
        self.vocal_clip_embeddings = []
        self.intrumental_clip_embeddings = []

    def __repr__(self):
        return (f"Song(path={self.path!r}, company={self.company!r}, debut_year={self.debut_year}, generation={self.generation!r}, "
                f"artist={self.artist!r}, name={self.song_name!r}), release_date={self.release_date}.\n")