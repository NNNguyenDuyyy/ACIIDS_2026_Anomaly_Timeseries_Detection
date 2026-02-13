# Haar Decomposition with Cross Attention for Time Series Anomaly Detection
ACIIDS_2026_Anomaly_Timeseries_Detection

Official implementation for our ACIIDS 2026 paper:
“Haar Decomposition with Cross Attention for Time Series Anomaly Detection.”
This work introduces an unsupervised framework combining fixed one-level Haar decomposition, shared-parameter bidirectional cross-attention, and branch-specific causal TCN decoders for robust time-series anomaly detection.

## Overview

Our model decomposes input windows into trend (low-frequency) and transient (high-frequency) components using the Haar wavelet.
A Mutual Attention mechanism enables both branches to exchange contextual information, while TCN decoders reconstruct each component.
Anomaly scores are derived from reconstruction errors.

![Our Architecture](https://github.com/user-attachments/assets/0cdfa1db-2b04-4fdc-b78b-4ec89449b38a)

## Reproducibility

All common hyperparameters are reported in **Table 1** of the paper (shared across datasets), and dataset-specific settings (e.g., `C`, `batch_size`, `epochs`) are reported in **Table 2**.  
To further improve reproducibility, we provide **pretrained checkpoints** and **step-by-step instructions** below to reproduce the reported results.
 
### Environment (Kaggle)
- Python: 3.12.12 (GCC 11.4.0)
- PyTorch: 2.8.0+cu126
- CUDA runtime: 12.6 (`torch.version.cuda`)
- GPU: Tesla P100-PCIE-16GB (compute capability 6.0)
- CUDA available: True
- We use a fixed random seed (**seed = 42**, as reported in Table 1). 

---

## Reproducibility Steps

### Step 1. Download Datasets & Checkpoints

We provide all datasets and pretrained checkpoints via Google Drive:

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
- Checkpoint: https://drive.google.com/drive/folders/16hfNGonOpRB1HtXUCawA1GRZQcLhPytj?usp=sharing

After downloading, extract them into a folder named datasets in the same directory as the notebook on Kaggle.

Expected structure:

    ├── 2dgesture/
      - 2DGesture_test.npy
      - 2DGesture_test_label.npy
      - 2DGesture_train.npy
    ├── checkpoint/aciids-2026-tsad-checkpoint/
      - best_model_2D_GESTURE.pth
      - best_model_ECG_A.pth
      - best_model_ECG_B.pth
      - best_model_ECG_C.pth
      - best_model_ECG_D.pth
      - best_model_ECG_E.pth
      - best_model_ECG_F.pth
      - best_model_MSL_C1.pth
      - best_model_PD.pth
      - best_model_SMD_11.pth
      - best_model_UCR_135.pth
      - best_model_UCR_136.pth
      - best_model_UCR_137.pth
      - best_model_UCR_138.pth
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

Upload **`aciids-2026-tsad-train-test.ipynb`** on Kaggle. This notebook contains the full training/testing pipeline.


### Step 3. Training & Testing (Main Entry Cell)

```python
if __name__ == "__main__":
    main(
        training=False,
        checkpoint="/kaggle/input/datasets/nguyenhuuduy04/checkpoint/aciids-2026-tsad-checkpoint/best_model_2D_GESTURE.pth",
        dataset="2DGesture",
        name_subset=None,
        data_path="/kaggle/input/datasets/nguyenhuuduy04/2dgesture",
        C=2,
        batch_size=64,
        epochs=10,     # use Table 2 for paper reproduction
        win_size=100,  # Table 1 (fixed)
        step=1         # stride for sliding windows
    )
```
**How to use this cell**

- Train from scratch: set training=True (checkpoint can be None).

- Test only / reproduce paper results: set training=False and provide a pretrained checkpoint.

- You can switch dataset and update data_path to match the dataset you want to run.

- Dataset-specific hyperparameters (C, batch_size, epochs) should follow Table 2 in the paper for strict reproduction.

- Common hyperparameters (e.g., win_size) follow Table 1.

**Recommended parameter mapping (paper → code)**

- Table 1 (common): win_size, model dimensions (d_model, heads, FFN), optimizer/lr, seed, etc.

- Table 2 (per dataset): C, batch_size, epochs, plus name_subset when applicable.

**Notes**

- Our model surpasses 12 state-of-the-art baselines on 5/6 benchmark datasets.

- Haar transform is fixed and applied per channel.
