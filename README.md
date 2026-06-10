# Universal EEG Epilepsy Detection via Evidential Multi-View De-Biasing

---

## 🧐 About

This repository provides a PyTorch implementation for the paper:

**"Universal EEG Epilepsy Detection via Evidential Multi-View De-Biasing"**

This work proposes a bias-guided Fisher-evidential multi-view learning framework for universal EEG epileptic seizure detection. The framework includes multi-view EEG feature construction, Fisher-evidential evidence estimation, low-evidence sample selection, biased branch training, and final debiased model training.

If you have any questions, feel free to contact **wednesque@gmail.com** — happy to discuss and exchange ideas!

If you find this work useful, please kindly cite our paper:

```bibtex
@inproceedings{wen2026universal,
  title={Universal EEG Epilepsy Detection via Evidential Multi-View De-Biasing},
  author={Wen, Ziqi and Xu, Cai and Zhao, Wanqing and Zhao, Jie and Zhao, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={32},
  pages={26849--26857},
  year={2026}
}
```

---

## 🎈 Usage

### Requirements

The code was developed with Python and PyTorch. We recommend the following environment:

```bash
python 3.8
numpy 1.23
pytorch 1.12
scikit-learn 1.2
tqdm
mne 0.23.4
scipy 1.1.0
matlab
```

You may install the main Python dependencies with:

```bash
pip install numpy scikit-learn tqdm mne scipy
```

Please install PyTorch according to your CUDA version from the official PyTorch website.

---

## ⭐ Preprocessing Procedure

### 1. Dataset Download

Please download EEG datasets from PhysioNet or other public EEG repositories, for example:

```text
https://physionet.org
```

Place the downloaded raw EEG files in:

```text
dataset/eeg/data/raw_data
```

The preprocessing scripts are provided to help users extract seizure segments, select channels, extract features, and construct multi-view data. Please adjust the scripts according to the dataset format, patient IDs, channel names, and seizure annotations used in your own experiments.

---

### 2. Channel Selection

Run:

```bash
python record.py
```

After running this script, you will obtain a record file containing seizure time slices extracted from the dataset.

**Note:** The extracted seizure time slices may include segments with different channel configurations. You can choose the required segments and channels according to your own experimental setting.

We also provide a pre-selected `record.txt` file to help users conveniently reproduce or customize the preprocessing procedure.

---

### 3. Channel Extraction

Run:

```bash
python channel.py
```

Please select the required patient seizure segments from `record.txt` and set them in the `segments` section of `channel.py`.

You may also modify the channel list according to the actual patient, EEG montage, and seizure segment selection used in your experiment.

---

### 4. Feature Extraction

Run the MATLAB feature extraction script:

```bash
cd dataset/eeg/preprocessing/ && matlab -nodesktop -nosplash -r preprocessing_data.m
```

This step extracts EEG domain features and generates the corresponding MATLAB feature files.

Please make sure the input and output paths in the MATLAB script are consistent with your local data directory.

---

### 5. Obtain Multi-view Data

Run:

```bash
python preprocessT.py
```

After running this script, you will obtain multi-view data files such as:

```text
train.pkl
valid.pkl
```

These files are used as the input data for model training and validation.

**Note:** We also provide an alternative preprocessing script in the `alternative` folder:

```bash
alternative/preprocess2.py
```

If `preprocessT.py` does not work properly for your data format or environment, you may try the alternative version.

---

### 6. Merge Multi-patient Data

To construct multi-patient training data, run:

```bash
python merge.py
```

This step merges data from multiple patients and generates the corresponding multi-patient training and validation files.

We also provide an alternative merging script in the `alternative` folder:

```bash
alternative/merge2.py
```

For debugging and reproduction, we recommend first training and testing the model on a single patient. After confirming that the single-patient pipeline works correctly, gradually merge multiple patients and evaluate the model under the multi-patient setting.

---

## 🚀 Training and Validation

After preprocessing and data merging, run:

```bash
python main.py
```

Before training, please carefully check the data paths in `main.py`, especially:

```python
train_path
valid_path
saving_path
```

Also check the corresponding input shape settings in `models.py` to make sure they match the generated multi-view data.

The training pipeline includes:

1. Pretraining a Fisher-evidential multi-view model.
2. Splitting samples according to evidence scores.
3. Training a biased model using low-evidence samples.
4. Training the final debiased model with HSIC-based bias mitigation.

The trained models will be saved to the path specified by `saving_path`.

---

## 📁 Suggested Project Structure

A typical project structure is as follows:

```text
BF-EML/
├── dataset/
│   └── eeg/
│       ├── data/
│       │   ├── raw_data/
│       │   └── domain_feature/
│       └── preprocessing/
│           └── preprocessing_data.m
├── alternative/
│   ├── preprocess2.py
│   └── merge2.py
├── record.py
├── channel.py
├── preprocessT.py
├── merge.py
├── data.py
├── models.py
├── main.py
└── README.md
```

---

## 🔧 Notes

- Please make sure the raw EEG files are placed in the correct directory before preprocessing.
- Please check the selected channels and seizure segments before feature extraction.
- If the generated data shape does not match the model input shape, please check both the preprocessing scripts and the view-specific modules in `models.py`.
- It is recommended to first run the full pipeline on one patient before conducting multi-patient experiments.
- Different EEG datasets may have different channel names, sampling rates, and annotation formats. Please modify the preprocessing scripts accordingly.
- The calibration fusion term in the released implementation is a simplified derivation of the calibration fusion algorithm described in the paper. Although the implementation form is simplified for training convenience, its practical objective is consistent with the formulation in the paper.

---

## 🙏 Acknowledgements

This codebase is built with reference to several excellent open-source projects. We sincerely thank the authors for their valuable contributions.

- **Multi-view fusion:**  
  https://github.com/Wednesque/RCML

- **Bias elimination:**  
  https://github.com/Wednesque/DEAR  
  https://github.com/Wednesque/rebias

- **Fisher evidential network:**  
  https://github.com/Wednesque/anedl  
  https://github.com/Wednesque/IEDL

---

## 📬 Contact

For questions, discussions, or suggestions, please contact:

```text
zqwenn@stu.xidian.edu.cn
```
