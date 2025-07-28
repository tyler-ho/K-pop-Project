import os
import torch
import numpy as np
from pathlib import Path
import warnings

def separate_track(input_file, output_dir, model_name="htdemucs_6s"):
    """
    Separate an audio track using Demucs with robust audio loading
    
    Args:
        input_file: Path to audio file
        output_dir: Directory to save separated stems
        model_name: Demucs model toconda use
    """
    # Convert to absolute path
    input_file = os.path.abspath(input_file)
    
    # Check file existence
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Audio file not found: {input_file}")
    
    print(f"Processing file: {input_file}")

    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    output_dir = output_dir + base_filename

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio using librosa instead of torchaudio
    print("Loading audio with librosa...")
    import librosa
    audio_data, sr = librosa.load(input_file, sr=None, mono=False)
    
    # Handle mono audio (convert to stereo)
    if len(audio_data.shape) == 1 or audio_data.shape[0] == 1:
        # If mono, duplicate to create stereo
        audio_data = np.stack([audio_data, audio_data])
        if len(audio_data.shape) == 3:
            audio_data = audio_data.squeeze(1)
    elif audio_data.shape[0] > 2:
        # If more than 2 channels, keep only the first two
        audio_data = audio_data[:2]
    
    # Convert to torch tensor
    wav = torch.tensor(audio_data, dtype=torch.float32)
    
    print(f"Audio loaded successfully. Shape: {wav.shape}, Sample rate: {sr}")
    
    # Import demucs modules
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    
    # Choose device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load model
    model = get_model(model_name)
    model.to(device)
    
    # Resample if needed
    if sr != model.samplerate:
        print(f"Resampling from {sr}Hz to {model.samplerate}Hz")
        # Use torch resample
        import torchaudio.functional as F
        wav = F.resample(wav, orig_freq=sr, new_freq=model.samplerate)
    
    # Add batch dimension
    wav = wav.unsqueeze(0)
    wav = wav.to(device)
    
    # Separate sources
    print("Separating sources...")
    with torch.no_grad():
        sources = apply_model(model, wav)
    
    # Remove batch dimension
    sources = sources.squeeze(0)
    
    # Save each source
    print("Saving separated sources...")
    for i, source_name in enumerate(model.sources):
        source_audio = sources[i]
        output_path = os.path.join(output_dir, f"{base_filename}_{source_name}.wav")
        
        # Use scipy to save wav files to avoid torchaudio issues
        from scipy.io import wavfile
        # Convert to numpy and adjust format/shape for scipy
        audio_numpy = source_audio.cpu().numpy()
        # Transpose to (samples, channels) as expected by scipy
        audio_numpy = audio_numpy.T
        
        print(f"Saving {source_name} to {output_path}")
        wavfile.write(output_path, model.samplerate, audio_numpy.astype(np.float32))
    
    print(f"Separation complete! Files saved to {output_dir}")
    return model.sources

# Example usage
if __name__ == "__main__":
    import pandas as pd
    data_path = 'DownloadViaChannel.xlsx'
    df = pd.read_excel(data_path)
    # for mac
    # for row in df.itertuples():
    #     input_dir = "audio_output_viachannel/" + row.company + '/' + str(row.generation) + '/' + row.group + '/'
    #     output_dir = "separated_stems/" + row.company + '/' + str(row.generation) + '/' + row.group + '/'
    #     for input_file in os.listdir(input_dir):
    #         if not os.path.exists(input_dir + input_file):
    #             separate_track(input_dir + input_file, output_dir)

    # for row in df.itertuples():
    #     input_dir = "audio_output_viachannel/" + row.company + '/' + str(row.generation) + '/' + row.group + '/'
    #     output_dir = os.path.join("separated_stems/", row.company, str(row.generation), row.group)
    #     for input_file in os.listdir(input_dir):
    #         separate_track(input_dir + input_file, output_dir)            
    
    import logging

    from datetime import datetime

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("file_processing.log", encoding="utf-8"),
            logging.StreamHandler()  # Also print to console
        ]
    )

    # Your processing code
    for row in df.itertuples():
        input_dir = os.path.join("audio_output_viachannel", row.company, str(row.generation), row.group)
        output_dir = os.path.join("separated_stems", row.company, str(row.generation), row.group)
        
        logging.info(f"Processing group: {row.company}/{row.generation}/{row.group}")
        
        for input_file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, input_file)
            output_file_path = os.path.join(output_dir, input_file)
            
            if not os.path.exists(output_file_path):
                logging.info(f"Processing: {file_path}")
                try:
                    separate_track(file_path, output_dir)
                    logging.info(f"Successfully processed: {file_path}")
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
            else:
                logging.info(f"Skipping (already exists): {file_path}")

    # Run the separation
    # separate_track(input_audio, output_directory)
