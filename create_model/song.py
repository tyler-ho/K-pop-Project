class Song:
    def __init__(self, path: str, company: str, generation: str, artist: str, name: str):
        self.path = path
        self.company = company
        self.generation = generation
        self.artist = artist
        self.name = name
        self.mfccs_path = ''

    def __repr__(self):
        return (f"Song(path={self.path!r}, company={self.company!r}, generation={self.generation!r}, "
                f"artist={self.artist!r}, name={self.name!r})")