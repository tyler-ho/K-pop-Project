# load_ext cuml.accel -- 
# load_ext cudf.pandas -- python -m cudf.pandas remove_duplicates_script.py

import torch, torchaudio, faiss
import os
import pickle
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pretrained Wav2Vec2 model on GPU
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(device).eval()

# Define functions for loading, preprocessing, embedding extraction
def load_and_preprocess(path, max_seconds=30):
    # Load MP3 file
    waveform, sample_rate = torchaudio.load(path)
    
    # Trim to max_seconds if longer
    max_samples = max_seconds * sample_rate
    if waveform.size(1) > max_samples:
        waveform = waveform[:, :max_samples]
    
    # Resample if needed
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)
        
    return waveform

def extract_embedding(waveform):
    waveform = waveform.to(device)
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)
        embedding = features[-1].mean(dim=1).squeeze().cpu().numpy()
    return embedding

def process_file(path):
    try:
        waveform = load_and_preprocess(path)
        embedding = extract_embedding(waveform)
        return path, embedding
    except Exception as e:
        print(f"Failed processing {path}: {e}")
        return path, None
    
# Parallel embedding extraction with ThreadPoolExecutor

def get_embeddings_parallel(filepaths, max_workers=4):
    embeddings = []
    valid_paths = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, path): path for path in filepaths}
        for future in as_completed(futures):
            path, emb = future.result()
            if emb is not None:
                embeddings.append(emb)
                valid_paths.append(path)
    embeddings = np.vstack(embeddings) if embeddings else np.array([])
    return embeddings, valid_paths

# Remove duplicates using FAISS GPU or CPU index

def remove_duplicates(embeddings, filepaths, threshold=0.8):
    d = embeddings.shape[1]
    if device.type == 'cuda':
        embeddings = embeddings.astype(np.float32)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(embeddings)
    k = 2
    D, I = index.search(embeddings, k)

    to_remove = set()
    for i in range(len(filepaths)):
        if i in to_remove:
            continue
        nearest_idx = I[i][1]
        dist = D[i][1]
        if dist < threshold:
            to_remove.add(nearest_idx)

    filtered_files = [f for i, f in enumerate(filepaths) if i not in to_remove]
    filtered_embeds = np.array([embeddings[i] for i in range(len(filepaths)) if i not in to_remove])
    return filtered_embeds, filtered_files, to_remove

if __name__ == '__main__':
    with open('create_model/farmshare_Songs.pkl', 'rb') as f:
        Songs = pickle.load(f)
    data = [vars(song) for song in Songs]
    df = pd.DataFrame(data)
    filepaths = df['path'].tolist()

    # Run parallel embedding extraction and duplicate removal

    max_workers = 16  # Adjust depending on your CPU cores

    embeddings, valid_paths = get_embeddings_parallel(filepaths, max_workers=max_workers)

    threshold = 0.8  # Tune threshold based on your data and results
    filtered_embeds, filtered_files, duplicates = remove_duplicates(embeddings, valid_paths, threshold)

    print(f"Original file count: {len(filepaths)}")
    print(f"Successfully processed: {len(valid_paths)}")
    print(f"Duplicates removed: {len(duplicates)}")
    print(f"Unique songs remaining: {len(filtered_files)}")

    with open('pickled_dupe_info/original_files.pkl', 'wb') as f:
        pickle.dump(filepaths, f)
    with open('pickled_dupe_info/valid_paths.pkl', 'wb') as f:
        pickle.dump(valid_paths, f)
    with open('pickled_dupe_info/duplicates.pkl', 'wb') as f:
        pickle.dump(duplicates, f)
    with open('pickled_dupe_info/unique_songs.pkl', 'wb') as f:
        pickle.dump(filtered_files, f)
