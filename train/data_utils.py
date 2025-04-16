import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from pydub import AudioSegment


def lie_list(dataset_path: str):
    """Generates file list and labels from split dataset directories.

    Args:
        dataset_path (str): Path to train/val/test split (e.g., 'E:/dataset/train').

    Returns:
        dict: {filename: label} mapping
        list: List of full file paths
    """
    dataset_path = Path(dataset_path)
    d_meta = {}
    file_list = []

    for label, subdir in enumerate(["Deceptive", "Truthful"]):  # Deceptive = 0, Truthful = 1
        dir_path = dataset_path / subdir
        mp3_files = sorted(list(dir_path.glob("*.mp3")))  # Get all MP3 files

        for file_path in mp3_files:
            key = file_path.stem  # File name without extension
            file_list.append(file_path)
            d_meta[key] = label  # Assign label

    return d_meta, file_list


def pad_random(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    """Randomly pads or truncates an audio signal to a fixed length.

    Args:
        x (np.ndarray): Audio waveform.
        max_len (int): Target length in samples.

    Returns:
        np.ndarray: Processed waveform.
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        start = np.random.randint(0, x_len - max_len + 1)
        return x[start: start + max_len]

    num_repeats = (max_len // x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


class Dataset_RLDD_train(Dataset):
    """Custom PyTorch Dataset for Training."""

    def __init__(self, file_list, labels):
        """
        Args:
            file_list (list): List of audio file paths.
            labels (dict): {filename: label} dictionary.
        """
        self.file_list = file_list
        self.labels = labels
        self.cut = 64600  # ~4 seconds

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        key = file_path.stem  # File name without extension

        # Load MP3 using pydub
        audio = AudioSegment.from_mp3(file_path)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        # Convert stereo to mono
        if audio.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)

        # Normalize to [-1, 1]
        samples = samples / np.max(np.abs(samples))

        # Apply random padding/truncation
        x_inp = Tensor(pad_random(samples, self.cut))
        
        # Get integer label
        label_int = self.labels[key]
        
        # Convert to one-hot encoded tensor
        # [1, 0] for Deceptive (0), [0, 1] for Truthful (1)
        y = torch.zeros(2)
        y[label_int] = 1.0
        
        return x_inp, key, y


class Dataset_RLDD_devNeval(Dataset):
    """Custom PyTorch Dataset for Validation & Evaluation (No Labels)."""

    def __init__(self, file_list):
        """
        Args:
            file_list (list): List of audio file paths.
        """
        self.file_list = file_list
        self.cut = 64600  # ~4 seconds

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        key = file_path.stem  # File name without extension

        # Load MP3 using pydub
        audio = AudioSegment.from_mp3(file_path)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        # Convert stereo to mono
        if audio.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)

        # Normalize to [-1, 1]
        samples = samples / np.max(np.abs(samples))

        # Apply padding/truncation
        x_inp = Tensor(pad_random(samples, self.cut))
        return x_inp, key, key
