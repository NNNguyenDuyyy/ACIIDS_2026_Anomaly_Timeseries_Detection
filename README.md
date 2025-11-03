# Haar Decomposition with Cross Attention for Time Series Anomaly Detection
ACIIDS_2026_Anomaly_Timeseries_Detection

Official implementation for our ACIIDS 2026 paper:
“Haar Decomposition with Cross Attention for Time Series Anomaly Detection.”
This work introduces an unsupervised framework combining fixed one-level Haar decomposition, shared-parameter bidirectional cross-attention, and branch-specific causal TCN decoders for robust time-series anomaly detection.

## Overview

Our model decomposes input windows into trend (low-frequency) and transient (high-frequency) components using the Haar wavelet.
A Mutual Attention mechanism enables both branches to exchange contextual information, while TCN decoders reconstruct each component.
Anomaly scores are derived from reconstruction errors.

## Reproduction Steps
### Step 1. Download Dataset

All datasets used in this paper are available via Google Drive:
- 2dgesture: https://drive.google.com/drive/folders/1XNTA2CVxOybKk2dD2akyeUanYnni-0ea?usp=sharing
- ecg-a-data: https://drive.google.com/drive/folders/1rZHBSGlHOtZqSUcEb2nzk_sH-erNFs5W?usp=sharing
- ecg-b-data: https://drive.google.com/drive/folders/1UPAbXMyPU1DPWTCiuylWz3MOjfC1mk-R?usp=sharing
- ecg-c-data: https://drive.google.com/drive/folders/19jzcGdIhbDUCuys4_vyPPCrZ-Nq5xXRt?usp=sharing
- ecg-d-data: https://drive.google.com/drive/folders/19LGHfOLR33sStUC4dz2jXk7rMBJqeRZN?usp=sharing
- ecg-e-data: https://drive.google.com/drive/folders/1QUGlF1p5MZKQm0w-DLdtlvQ5Mz4nn11U?usp=sharing
- ecg-f-data: https://drive.google.com/drive/folders/17vhvtCcaS76sm0Byw6r_z56sGtd5GHKg?usp=sharing
- msl-c1: https://drive.google.com/drive/folders/1rr9jfAEp7TtuGooRaz_CqoyrsEn7b0wF?usp=sharing
- pd: https://drive.google.com/drive/folders/1W83f6I2yKDNAaAFiVM_6p5R1vCyh9h6R?usp=sharing
- smd-11: https://drive.google.com/drive/folders/1YeBPNU-qhW94x9VHRit5vnuHXrmwhQm8?usp=sharing
- ucr-135: https://drive.google.com/drive/folders/1vNEF9i9_-0Kjmgqht_wIusAsBvEj0kz-?usp=sharing
- ucr-136: https://drive.google.com/drive/folders/13bL9EscTAlnmrs7w-SBXzTZJZnXU5kYs?usp=sharing
- ucr-137: https://drive.google.com/drive/folders/1J8iSES6r-4llOMYtCw4aclY9bBquTtvO?usp=sharing
- ucr-138: https://drive.google.com/drive/folders/14MIk2troiMnzT_JMWC7DQyrg3wQjvyIT?usp=sharing

After downloading, extract them into a folder named datasets in the same directory as the notebook on Kaggle.


    ├── 2dgesture/
      - 2DGesture_test.npy
      - 2DGesture_test_label.npy
      - 2DGesture_train.npy
    ├── ecg-a-data/
      - ECG_test.npy
      - ECG_test_label.npy
      - ECG_train.npy
    ├── ecg-b-data/
      - ECG_test.npy
      - ECG_test_label.npy
      - ECG_train.npy
    ├── ecg-c-data/
      - ECG_test.npy
      - ECG_test_label.npy
      - ECG_train.npy
    ├── ecg-d-data/
      - ECG_test.npy
      - ECG_test_label.npy
      - ECG_train.npy
    ├── ecg-e-data/
      - ECG_test.npy
      - ECG_test_label.npy
      - ECG_train.npy
    ├── ecg-f-data/
      - ECG_test.npy
      - ECG_test_label.npy
      - ECG_train.npy
    |── msl-c1/
      - C-1_labels.npy
      - C-1_test.npy
      - C-1_train.npy
    |── pd-test/
      - power_data.pkl
    |── pd-train/
      - power_data.pkl 
    |── smd-11/
      - machine-1-1_labels.npy
      - machine-1-1_test.npy
      - machine-1-1_train.npy
    ├── ucr-135/
      - UCR_test.npy
      - UCR_test_label.npy
      - UCR_train.npy
    ├── ucr-136/
      - UCR_test.npy
      - UCR_test_label.npy
      - UCR_train.npy
    ├── ucr-137/
      - UCR_test.npy
      - UCR_test_label.npy
      - UCR_train.npy
    └── ucr-138/
      - UCR_test.npy
      - UCR_test_label.npy
      - UCR_train.npy

### Step 2. Upload Notebook on Kaggle

Upload **aciids-2026-tsad.ipynb** on Kaggle


### Step 3. Run Notebook

Open and execute all cells in aciids-2026-tsad.ipynb:

Modify hyperparameters:

data_path = "/kaggle/input/ucr-135" # for example  

batch_size = 32

win_size = 100

step = 1

C = 1

num_epochs = 10

Our model surpasses 12 state-of-the-art baselines on 5 out of 6 benchmark datasets, demonstrating strong robustness and generalization.
🧪 Notes

Experiments conducted on Kaggle GPU (Tesla P100).

Haar transform is fixed and applied per channel.
