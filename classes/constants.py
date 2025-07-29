from file_processing.create_clip_embeddings import get_feature_labels

# Ignore/remove copied songs in other languages, instrumentals, remixes, and
# concert recordings.
BANNED_WORDS = ['korean', 'kor', 'chinese', 'japanese', 'jpn', 'thai',
                'inst', 'karaoke', 'version', 'mix', '~',
                'live', 'tour']
FEATURE_LABELS = get_feature_labels() # labels for librosa features
N_JOBS = 1 # number of threads to create; used for multi-threading
RANDOM_STATE = 42 # random state; used for reproducibility

# Embedding Types
VCE = 'vocal_clip_embeddings'
ICE = 'intrumental_clip_embeddings'
