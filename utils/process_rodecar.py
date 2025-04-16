import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_directory_structure():
    """Create the required directory structure for the dataset."""
    base_dir = "romanian_dataset_24kb"
    splits = ["train", "val", "test"]
    classes = ["Deceptive", "Truthful"]
    
    # Create base directory first
    os.makedirs(base_dir, exist_ok=True)
    
    # Create all subdirectories
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(base_dir, split, class_name), exist_ok=True)
            # Also create temporary directories for initial processing
            os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)

def process_audio_file(audio_path, annotation_path, output_base_dir, file_counter):
    """Process a single audio file according to its annotation."""
    # Read the annotation CSV
    df = pd.read_csv(annotation_path)
    
    # Load the audio file
    audio = AudioSegment.from_wav(audio_path)
    
    processed_segments = []
    
    # Process each segment
    for _, row in df.iterrows():
        # Skip if the speaker is TM (prosecutor)
        if row['speaker'] == 'TM':
            continue
            
        # Get start and stop times in milliseconds
        start_ms = int(row['startTime'] * 1000)
        stop_ms = int(row['stopTime'] * 1000)
        
        # Extract the segment
        segment = audio[start_ms:stop_ms]
        
        # Determine if the segment is deceptive or truthful
        is_deceptive = row['annotation'] == 'F'
        
        # Create the output filename
        class_name = "Deceptive" if is_deceptive else "Truthful"
        counter = file_counter[class_name]
        filename = f"trial_{'lie' if is_deceptive else 'truth'}_{counter:03d}.mp3"
        
        # Save the segment with low bitrate (24kbps) for smaller file size
        output_path = os.path.join(output_base_dir, class_name, filename)
        segment.export(output_path, format="mp3", bitrate="24k")
        
        processed_segments.append({
            'filename': filename,
            'class': class_name,
            'duration': len(segment) / 1000.0  # duration in seconds
        })
        
        file_counter[class_name] += 1
    
    return processed_segments

def process_single_file_for_verification(file_name):
    """Process only a single file without randomization for verification purposes."""
    # Create verification directory
    base_dir = "romanian_dataset_24kb"
    verification_dir = os.path.join(base_dir, f"verification_{file_name}")
    classes = ["Deceptive", "Truthful"]
    
    # Create verification directories
    os.makedirs(verification_dir, exist_ok=True)
    for class_name in classes:
        os.makedirs(os.path.join(verification_dir, class_name), exist_ok=True)
    
    # Paths
    annotation_dir = "raw_data/RODeCAR/RODeCAR/Annotation"
    audio_dir = "raw_data/RODeCAR/RODeCAR/Files_WAV"
    
    csv_file = f"{file_name}.csv"
    audio_file = f"{file_name}.wav"
    
    audio_path = os.path.join(audio_dir, audio_file)
    annotation_path = os.path.join(annotation_dir, csv_file)
    
    if not os.path.exists(audio_path) or not os.path.exists(annotation_path):
        print(f"Error: Could not find {audio_file} or {csv_file}")
        return
    
    # Process the file
    print(f"Processing {audio_file} for verification...")
    
    # Read the annotation CSV
    df = pd.read_csv(annotation_path)
    
    # Load the audio file
    audio = AudioSegment.from_wav(audio_path)
    
    processed_segments = []
    
    # Process each segment with original index for verification
    for idx, row in df.iterrows():
        # Skip if the speaker is TM (prosecutor)
        if row['speaker'] == 'TM':
            continue
            
        # Get start and stop times in milliseconds
        start_ms = int(row['startTime'] * 1000)
        stop_ms = int(row['stopTime'] * 1000)
        
        # Extract the segment
        segment = audio[start_ms:stop_ms]
        
        # Determine if the segment is deceptive or truthful
        is_deceptive = row['annotation'] == 'F'
        
        # Create the output filename with original segment index for verification
        class_name = "Deceptive" if is_deceptive else "Truthful"
        filename = f"segment_{idx:03d}_{'lie' if is_deceptive else 'truth'}.mp3"
        
        # Save the segment with low bitrate (24kbps) for smaller file size
        output_path = os.path.join(verification_dir, class_name, filename)
        segment.export(output_path, format="mp3", bitrate="24k")
        
        processed_segments.append({
            'original_index': idx,
            'filename': filename,
            'class': class_name,
            'duration': len(segment) / 1000.0,  # duration in seconds
            'start_time': row['startTime'],
            'stop_time': row['stopTime'],
            'speaker': row['speaker'],
            'annotation': row['annotation'],
            'transcription': row.get('transcription', '') if 'transcription' in row else ''
        })
    
    # Create a CSV file with segment information for verification
    df_segments = pd.DataFrame(processed_segments)
    df_segments.to_csv(os.path.join(verification_dir, "segment_info.csv"), index=False)
    
    # Print statistics
    print("\nVerification Dataset Statistics:")
    print(f"Total segments: {len(processed_segments)}")
    print("\nClass distribution:")
    class_counts = df_segments['class'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} segments ({count/len(processed_segments)*100:.1f}%)")
    
    print(f"\nVerification data saved to: {verification_dir}")
    print("Segment information saved to: segment_info.csv")

def analyze_duration_distribution():
    """Analyze the distribution of audio segment durations in the dataset."""
    base_dir = "romanian_dataset_24kb"
    annotation_dir = "raw_data/RODeCAR/RODeCAR/Annotation"
    
    # Collect all segment durations from annotation files
    csv_files = [f for f in os.listdir(annotation_dir) if f.endswith('.csv')]
    
    all_durations = []
    truthful_durations = []
    deceptive_durations = []
    
    print("Analyzing duration distribution from annotation files...")
    for csv_file in tqdm(csv_files, desc="Processing annotation files"):
        annotation_path = os.path.join(annotation_dir, csv_file)
        df = pd.read_csv(annotation_path)
        
        # Check if 'speaker' column exists before filtering
        if 'speaker' in df.columns:
            # Filter out prosecutor segments
            df = df[df['speaker'] != 'TM']
        
        # Check if 'duration' column exists
        if 'duration' not in df.columns:
            print(f"Warning: 'duration' column not found in {csv_file}, skipping...")
            continue
            
        # Calculate durations
        durations = df['duration'].values
        all_durations.extend(durations)
        
        # Check if 'annotation' column exists
        if 'annotation' in df.columns:
            # Separate truthful and deceptive
            truthful_durations.extend(df[df['annotation'] == 'T']['duration'].values)
            deceptive_durations.extend(df[df['annotation'] == 'F']['duration'].values)
        else:
            print(f"Warning: 'annotation' column not found in {csv_file}, skipping truthful/deceptive separation...")
    
    # Collect all segment durations from processed files
    processed_durations = []
    
    print("\nAnalyzing duration distribution from processed audio files...")
    splits = ["train", "val", "test"]
    classes = ["Deceptive", "Truthful"]
    
    for split in splits:
        for class_name in classes:
            dir_path = os.path.join(base_dir, split, class_name)
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    if filename.endswith('.mp3'):
                        file_path = os.path.join(dir_path, filename)
                        try:
                            audio = AudioSegment.from_mp3(file_path)
                            duration = len(audio) / 1000.0  # duration in seconds
                            processed_durations.append(duration)
                        except Exception as e:
                            print(f"Error loading file {file_path}: {e}")
    
    # Create bins for duration ranges
    duration_bins = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 60, float('inf')]
    bin_labels = ['<1s', '1-2s', '2-3s', '3-4s', '4-5s', '5-10s', '10-15s', '15-20s', '20-30s', '30-60s', '>60s']
    
    # Count durations in each bin
    annotation_hist, _ = np.histogram(all_durations, bins=duration_bins)
    processed_hist, _ = np.histogram(processed_durations, bins=duration_bins)
    truthful_hist, _ = np.histogram(truthful_durations, bins=duration_bins)
    deceptive_hist, _ = np.histogram(deceptive_durations, bins=duration_bins)
    
    # Create DataFrame for display
    df_duration = pd.DataFrame({
        'Duration Range': bin_labels,
        'Annotation Count': annotation_hist,
        'Annotation %': (annotation_hist / len(all_durations) * 100) if all_durations else 0,
        'Processed Count': processed_hist,
        'Processed %': (processed_hist / len(processed_durations) * 100) if processed_durations else 0,
        'Truthful Count': truthful_hist,
        'Truthful %': (truthful_hist / len(truthful_durations) * 100) if truthful_durations else 0,
        'Deceptive Count': deceptive_hist,
        'Deceptive %': (deceptive_hist / len(deceptive_durations) * 100) if deceptive_durations else 0
    })
    
    # Print statistics
    print("\nDuration Distribution Statistics:")
    print(f"Total segments from annotations: {len(all_durations)}")
    print(f"Total processed segments: {len(processed_durations)}")
    print(f"Total truthful segments: {len(truthful_durations)}")
    print(f"Total deceptive segments: {len(deceptive_durations)}")
    
    # Display duration distribution
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print("\nDuration Distribution:")
    print(df_duration.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    # Save the duration distribution to CSV
    output_path = os.path.join(base_dir, "duration_distribution.csv")
    df_duration.to_csv(output_path, index=False)
    print(f"\nDuration distribution saved to: {output_path}")
    
    # Create plots if matplotlib is available
    try:
        # Plot duration distribution
        plt.figure(figsize=(12, 6))
        
        # Plot annotation durations
        plt.subplot(1, 2, 1)
        plt.bar(bin_labels, annotation_hist, alpha=0.7)
        plt.title('Duration Distribution (Annotations)')
        plt.xlabel('Duration Range')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Plot truthful vs deceptive durations
        plt.subplot(1, 2, 2)
        width = 0.35
        x = np.arange(len(bin_labels))
        plt.bar(x - width/2, truthful_hist, width, label='Truthful', alpha=0.7)
        plt.bar(x + width/2, deceptive_hist, width, label='Deceptive', alpha=0.7)
        plt.title('Truthful vs Deceptive Duration Distribution')
        plt.xlabel('Duration Range')
        plt.ylabel('Count')
        plt.xticks(x, bin_labels, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(base_dir, "duration_distribution.png"))
        print(f"Duration distribution plot saved to: {os.path.join(base_dir, 'duration_distribution.png')}")
    except Exception as e:
        print(f"Could not create plots: {e}")

def main_create_dataset():
    # Create directory structure
    create_directory_structure()
    
    # Initialize counters for file naming
    file_counter = {"Deceptive": 1, "Truthful": 1}
    
    # Paths
    annotation_dir = "raw_data/RODeCAR/RODeCAR/Annotation"
    audio_dir = "raw_data/RODeCAR/RODeCAR/Files_WAV"
    output_base_dir = "romanian_dataset_24kb"
    
    # Get all CSV files (excluding RODeCAR_bff.xlsx and silence_list.xlsx)
    csv_files = [f for f in os.listdir(annotation_dir) if f.endswith('.csv')]
    
    # Process all files and collect segments
    all_segments = []
    for csv_file in tqdm(csv_files, desc="Processing audio files"):
        audio_file = csv_file.replace('.csv', '.wav')
        audio_path = os.path.join(audio_dir, audio_file)
        annotation_path = os.path.join(annotation_dir, csv_file)
        
        if os.path.exists(audio_path):
            segments = process_audio_file(audio_path, annotation_path, output_base_dir, file_counter)
            all_segments.extend(segments)
    
    # Convert to DataFrame for easier manipulation
    df_segments = pd.DataFrame(all_segments)
    
    # Shuffle the segments
    df_segments = df_segments.sample(frac=1, random_state=42)
    
    # Calculate split sizes
    total_segments = len(df_segments)
    train_size = int(0.75 * total_segments)
    val_size = int(0.10 * total_segments)
    test_size = total_segments - train_size - val_size
    
    # Split the data
    train_data = df_segments[:train_size]
    val_data = df_segments[train_size:train_size + val_size]
    test_data = df_segments[train_size + val_size:]
    
    # Move files to their respective directories
    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        for _, row in data.iterrows():
            src_path = os.path.join(output_base_dir, row['class'], row['filename'])
            dst_path = os.path.join(output_base_dir, split, row['class'], row['filename'])
            os.rename(src_path, dst_path)
    
    # Clean up temporary directories
    for class_name in ["Deceptive", "Truthful"]:
        temp_dir = os.path.join(output_base_dir, class_name)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total segments: {total_segments}")
    print("\nTrain set:")
    print(train_data['class'].value_counts())
    print("\nValidation set:")
    print(val_data['class'].value_counts())
    print("\nTest set:")
    print(test_data['class'].value_counts())

# if __name__ == "__main__":
#     main() 

# process_single_file_for_verification("F1_2_Q2")
# analyze_duration_distribution()
main_create_dataset()