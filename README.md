# Speech Understanding Course Project

![Poster](https://github.com/user-attachments/assets/5ad8d773-2b08-455e-bd64-cbba9efce473)


# Graph Attention Network (AASIST)-Based Speech Lie Detection

This repository contains the implementation of a speech-based lie detection system leveraging the AASIST (Audio Anti-Spoofing System) framework. The model is designed to analyze speech recordings and classify them as either truthful or deceptive based on acoustic patterns.

## Project Overview

Deception detection using speech processing has significant applications in security, forensics, and psychology. This project adapts the AASIST framework, originally developed for audio anti-spoofing, to detect deception in speech across multiple languages. We evaluate the system on both English and Romanian datasets.

## Datasets

The system is trained and evaluated on two main datasets:

1. **Real-life Deception Detection Dataset (RLDD)**: 
   - English audio recordings from courtroom trials
   - Contains truthful and deceptive testimonies in uncontrolled acoustic environments

2. **Romanian Deva Criminal Investigation Audio Recordings Dataset (RODeCAR)**:
   - Romanian audio recordings from criminal investigations
   - Provides cross-lingual evaluation capability

## Key Features

- Adaptation of AASIST framework for binary lie detection classification
- Cross-dataset and cross-lingual evaluation
- Comprehensive performance metrics (Accuracy, Precision, Recall, F1, EER, AUC)
- Performance visualization tools (ROC curves, PR curves, confusion matrices)

## Repository Structure

```
├── LICENSE                         # Project license
├── models/                         # Model architecture definitions
│   ├── AASIST.py                   # AASIST model implementation
│   └── AASIST.pth                  # Pre-trained model weights
├── config/                         # Configuration files
├── utils/                          # Utility functions for training
├── notebooks/                      # Jupyter notebooks containing experiments and plots
├── data/                           # Dataset directory containing RLDD and Rodecar
├── results/                        # Results saved from the best experiments (metrics and models)
├── experiment_dump/                # Experiment outputs
├── train/                          # Training scripts
├── visualizations/                 # Generated architecture visualizations
│   ├── aasist_diagram.png          # Architecture diagram
│   └── visualize_aasist_diagram.py # Diagram generation script
└── README.md                       # This file
```


## Requirements

- Python 3.7+
- PyTorch 1.8+
- TensorBoard
- scikit-learn
- numpy
- pandas
- matplotlib
- [Other dependencies as needed]

## Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare datasets:
   - Download RLDD and RODeCAR datasets
   - Place them in the appropriate directories (dataset/, romanian_dataset/)
   - Run preprocessing:
     ```
     python process_rodecar.py
     ```

## Usage

### Training

To train the model on the RLDD dataset:

```
python main.py --train --dataset RLDD --epochs 75 --batch_size 8 --config config/aasist_rldd.conf
```


### Evaluation

To evaluate a trained model:

```
python main.py --eval --dataset [RLDD/RODeCAR] --model_path [path_to_saved_model] --config [config_file]
```


### Generating Performance Metrics and Plots
```
python evaluation.py --score_path [path_to_scores] --output_dir [output_directory]
```

## Results

### RLDD Dataset
- Accuracy: 63.16%
- Precision: 75.00%
- Recall: 33.33%
- F1 Score: 46.15%
- EER: 52.78%
- ROC AUC: 0.567

### RODeCAR Dataset
- Accuracy: 81.82%
- Precision: 80.98%
- Recall: 85.53%
- F1 Score: 83.19%
- EER: 18.52%
- ROC AUC: 0.886

The model demonstrated significantly better performance on the RODeCAR dataset compared to RLDD, suggesting sensitivity to dataset characteristics, language factors, and sample size differences.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- Shyam Sathvik (b22ee036@iitj.ac.in)
- Neermita Bhattacharya (b22cs092@iitj.ac.in)

## References

- https://arxiv.org/pdf/2110.01200
- https://public.websites.umich.edu/~zmohamed/resources.html
- https://ieeexplore.ieee.org/document/8906542

