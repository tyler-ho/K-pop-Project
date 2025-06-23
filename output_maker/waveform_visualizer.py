import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
from tqdm import tqdm  # for progress bar (optional)
import warnings
warnings.filterwarnings('ignore')

class BatchWaveformGenerator:
    """
    Process all audio files in a folder and generate waveform images
    """
    
    def __init__(self, input_folder, output_folder=None, supported_formats=None):
        """
        Initialize the batch processor
        
        Parameters:
        input_folder: Path to folder containing audio files
        output_folder: Path to save images (if None, creates 'waveforms' subfolder)
        supported_formats: List of audio formats to process
        """
        self.input_folder = Path(input_folder)
        
        if output_folder is None:
            self.output_folder = self.input_folder / 'waveforms'
        else:
            self.output_folder = Path(output_folder)
            
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(exist_ok=True)
        
        # Default supported audio formats
        if supported_formats is None:
            self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', 
                                    '.opus', '.aac', '.wma', '.aiff', '.au'}
        else:
            self.supported_formats = set(supported_formats)
    
    def get_audio_files(self):
        """Get all audio files in the input folder"""
        audio_files = []
        
        for file_path in self.input_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                audio_files.append(file_path)
                
        return sorted(audio_files)
    
    def create_waveform(self, audio_path, style='basic', **kwargs):
        """Create waveform for a single audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(str(audio_path), sr=None)
            
            # Generate output filename
            output_name = f"{audio_path.stem}_waveform.png"
            output_path = self.output_folder / output_name
            
            if style == 'basic':
                self._create_basic_waveform(y, sr, output_path, **kwargs)
            elif style == 'styled':
                self._create_styled_waveform(y, sr, output_path, **kwargs)
            elif style == 'bars':
                self._create_bars_waveform(y, sr, output_path, **kwargs)
            elif style == 'minimal':
                self._create_minimal_waveform(y, sr, output_path, **kwargs)
                
            return True, output_path
            
        except Exception as e:
            print(f"Error processing {audio_path.name}: {str(e)}")
            return False, None
    
    def _create_basic_waveform(self, y, sr, output_path, figsize=(12, 4), 
                               color='#1f77b4', dpi=150):
        """Create basic waveform visualization"""
        plt.figure(figsize=figsize)
        
        # Create time axis
        duration = len(y) / sr
        time = np.linspace(0, duration, len(y))
        
        # Plot waveform
        plt.plot(time, y, color=color, linewidth=0.5)
        plt.fill_between(time, y, alpha=0.3, color=color)
        
        # Styling
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Waveform: {output_path.stem.replace("_waveform", "")}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def _create_styled_waveform(self, y, sr, output_path, figsize=(14, 6),
                               color='#00ff41', bg_color='#0a0a0a', dpi=150):
        """Create styled waveform visualization"""
        # Downsample for visualization
        hop_length = max(1, len(y) // 4000)
        y_resample = y[::hop_length]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Create time axis
        duration = len(y) / sr
        time = np.linspace(0, duration, len(y_resample))
        
        # Plot waveform with mirror effect
        ax.plot(time, y_resample, color=color, linewidth=0.7, alpha=0.9)
        ax.plot(time, -y_resample * 0.5, color=color, linewidth=0.5, alpha=0.4)
        
        # Styling
        ax.set_xlim(0, duration)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor=bg_color, edgecolor='none')
        plt.close()
    
    def _create_bars_waveform(self, y, sr, output_path, figsize=(12, 6), 
                             num_bars=100, colormap='viridis', dpi=150):
        """Create bar-style waveform visualization"""
        # Calculate RMS energy for each segment
        hop_length = len(y) // num_bars
        rms_values = []
        
        for i in range(num_bars):
            start = i * hop_length
            end = min((i + 1) * hop_length, len(y))
            segment = y[start:end]
            rms = np.sqrt(np.mean(segment**2))
            rms_values.append(rms)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bars with gradient
        x = np.arange(num_bars)
        bars = ax.bar(x, rms_values, width=0.8)
        
        # Apply colormap
        colors = plt.cm.get_cmap(colormap)(np.array(rms_values) / max(rms_values))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Styling
        ax.set_xlim(-0.5, num_bars - 0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title(f'Waveform: {output_path.stem.replace("_waveform", "")}')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def _create_minimal_waveform(self, y, sr, output_path, figsize=(12, 3),
                                color='black', dpi=150):
        """Create minimal waveform visualization"""
        # Heavy downsampling for smooth appearance
        hop_length = max(1, len(y) // 2000)
        y_resample = y[::hop_length]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create time axis
        duration = len(y) / sr
        time = np.linspace(0, duration, len(y_resample))
        
        # Plot only the outline
        ax.fill_between(time, y_resample, alpha=1.0, color=color, linewidth=0)
        ax.fill_between(time, -y_resample, alpha=1.0, color=color, linewidth=0)
        
        # Remove all axes and labels
        ax.set_xlim(0, duration)
        ax.set_ylim(-1.1, 1.1)
        ax.axis('off')
        
        # Remove margins
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def process_all(self, style='basic', show_progress=True, **kwargs):
        """
        Process all audio files in the folder
        
        Parameters:
        style: 'basic', 'styled', 'bars', or 'minimal'
        show_progress: Show progress bar
        **kwargs: Additional parameters for the specific style
        """
        audio_files = self.get_audio_files()
        
        if not audio_files:
            print(f"No audio files found in {self.input_folder}")
            return
        
        print(f"Found {len(audio_files)} audio files")
        print(f"Output folder: {self.output_folder}")
        
        successful = 0
        failed = 0
        
        # Process files with optional progress bar
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_files, desc="Processing files")
            except ImportError:
                print("Install tqdm for progress bar: pip install tqdm")
                iterator = audio_files
        else:
            iterator = audio_files
        
        for audio_file in iterator:
            success, output_path = self.create_waveform(audio_file, style, **kwargs)
            
            if success:
                successful += 1
                if not show_progress:
                    print(f"✓ Processed: {audio_file.name}")
            else:
                failed += 1
                if not show_progress:
                    print(f"✗ Failed: {audio_file.name}")
        
        print(f"\nProcessing complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

def batch_process_folder(input_folder, output_folder=None, style='basic', **kwargs):
    """
    Convenience function to process all audio files in a folder
    
    Parameters:
    input_folder: Path to folder with audio files
    output_folder: Where to save images (optional)
    style: 'basic', 'styled', 'bars', or 'minimal'
    **kwargs: Style-specific parameters
    """
    processor = BatchWaveformGenerator(input_folder, output_folder)
    processor.process_all(style=style, **kwargs)

# Example usage with different scenarios
if __name__ == "__main__":
    # Example 1: Basic processing of all files in a folder
    input_folder = "separated_stems/JYP/4/Stray Kids/CIRCUS"
    
    # Create basic waveforms
    batch_process_folder(input_folder, style='basic')
    
    # Example 2: Create styled waveforms with custom colors
    batch_process_folder(
        input_folder,
        output_folder="circus_visualized",
        style='styled',
        color='#ff6b6b',
        bg_color='#1a1a1a'
    )
    
    # Example 3: Create bar visualizations
    batch_process_folder(
        input_folder,
        style='bars',
        num_bars=150,
        colormap='plasma'
    )
    
    # Example 4: Process specific file types only
    processor = BatchWaveformGenerator(
        input_folder,
        supported_formats=['.wav', '.mp3']
    )
    processor.process_all(style='minimal')
    
    # Example 5: Process with custom settings for each style
    processor = BatchWaveformGenerator(input_folder)
    
    # You can also process the same files