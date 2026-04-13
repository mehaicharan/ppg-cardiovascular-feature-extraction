# Wrist vs Finger PPG — Multi-Metric Cardiovascular Monitoring Pipeline


This repository contains the complete Python signal-processing pipeline developed for a controlled, simultaneous comparison of wrist (MAXM86146) and finger (AFE4490) photoplethysmography (PPG) signals. The pipeline covers preprocessing, two-step synchronisation, signal quality assessment, protocol-based window selection, and feature extraction for six cardiovascular metrics: heart rate (HR), heart rate variability (HRV), breathing rate (BR), peripheral oxygen saturation (SpO₂), stiffness index (SI) surrogate, and reflection index (RI). Data were collected from 20 healthy participants (10 younger, ages 18–25; 10 older, ages 45+) under resting and paced-breathing conditions.

This repository includes the code developed for the BEng thesis:

**Comparative Assessment of Wrist and Finger Photoplethysmography for Multi-Metric Cardiovascular Monitoring**  
**Mehai Singh Charan**  
BEng Electronics and Electrical Engineering  
The University of Edinburgh, 2026


![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Code-yellow)


## Repository Structure

```text
ppg-pipeline/
│
├── data/                         # Raw CSV recordings (NOT included)
│   ├── finger/
│   └── wrist/
│
├── src/
│   ├── preprocessing.py          # Inversion, timestamping, hybrid filtering 
│   ├── synchronisation.py        # Two-step thump + cross-correlation sync 
│   ├── quality_assessment.py         # Window quality scoring (Q-score) 
│   ├── window_selection.py       # Protocol-based optimal window selection 
│   ├── hr_hrv_extraction.py                 # HR and HRV extraction
│   ├── br_extraction.py         # BR extraction via Welch PSD
│   ├── spo2_extraction.py                   # SpO₂ (AC/DC ratio, Hampel filtering)
│   └── vascular_indices_extraction.py       # RI and SI surrogate (APG fallback)
│
├── thesis/
│   └── final.thesis.pdf  # End-to-end explaination (will upload by 15th June)
│
├
└── README.md
```

## Pipeline Overview

End-to-end processing:

```text
Raw CSVs
 (finger: AFE4490 @ 500 Hz,
  wrist: MAXM86146 @ variable Hz)
        │
        ▼
 Preprocessing[1]
    - Signal inversion
    - Timestamp construction & resampling to 500 Hz
    - Wavelet-based DC removal + Butterworth bandpass (0.5–4.0 Hz)
        │
        ▼
 Synchronisation
    - Initial “thump” artefact alignment (first 15 s)
    - Cross-correlation refinement (10–30 s window)
    - Common 500 Hz master time axis
        │
        ▼
 Signal Quality Assessment
    - 10 s sliding window (50% overlap)
    - Composite Q-score and validity mask per site
    - Joint validity mask = logical AND (finger, wrist)
        │
        ▼
 Protocol-Based Window Selection
    - Best 90 s resting window
    - Best 90 s (or 120 s) paced-breathing window
        │
        ▼
 Feature Extraction
    - HR, HRV (SDNN, RMSSD, pNN50)
    - BR (Welch PSD, 0.083–0.5 Hz)
    - SpO₂ (AC/DC ratio, calibration curve, Hampel filter)
    - RI & SI surrogate (PPG + APG morphology, diastolic fallback)
```
---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/<your-username>/ppg-pipeline.git
cd ppg-pipeline
pip install -r requirements.txt
```

Core dependencies (add versions in `requirements.txt`):

| Package      | Purpose                                             |
|-------------|-----------------------------------------------------|
| `numpy`     | Array operations, RR-interval arithmetic            |
| `scipy`     | Butterworth filter, cross-correlation, Welch PSD    |
| `pandas`    | CSV loading, timestamp parsing                      |
| `PyWavelets`| db4 wavelet DC removal                              |

---

## Data

Raw PPG recordings from human participants are **not included** in this repository because of ethics and privacy constraints (University of Edinburgh Research Ethics, GDPR, UK Data Protection Act 2018).

The code expects CSV files with the following structure:

| File pattern       | Sensor    | Key columns (examples)                                 |
|--------------------|-----------|--------------------------------------------------------|
| `finger_XX.csv`    | AFE4490   | `Ch LED 2 RED`, `Ch LED 2 RED AMBIENT`, ...           |
| `wrist_XX.csv`     | MAXM86146 | `LEDC1` (or equivalent LED channel), `timestamp` (ms) |

Configure input paths and target sample rate at the top of `preprocessing.py`:

```python
FINGER_FILE_PATH      = "data/finger/participant_01_finger.csv"
WRIST_FILE_PATH       = "data/wrist/participant_01_wrist.csv"
TARGET_SAMPLE_RATE_HZ = 500
```
To run the notebook without sharing participant data, you can:

- Use a small anonymised subset of your recordings, or  
- Simulate simple PPG-like example data for demonstration purposes only.

---
## Key Methods (Brief Description)

### Hybrid Filtering

Each PPG signal passes through a **two-stage filter**:

1. **Wavelet DC removal**  
   A db4 discrete wavelet transform (DWT) is applied; approximation coefficients are zeroed and the signal is reconstructed to remove slow-varying baseline components.

2. **Zero-phase bandpass**  
   A 4th‑order Butterworth bandpass (0.5–4.0 Hz) is applied with `filtfilt` to remove out-of-band noise without phase distortion across the cardiac band.

### Two-Step Synchronisation

- **Initial alignment**: detect the synchronisation “thump” artefact in the first 15 s of both wrist and finger traces, and compute an initial time shift.
- **Refinement**: perform discrete cross-correlation on a 10–30 s segment to refine the lag and shift one signal so that both are phase-locked on a common 500 Hz time axis.

### Signal Quality Assessment

- A 10 s sliding window with 50% overlap evaluates signal quality.
- For each window, peaks and troughs are detected, and a composite quality score \( Q \) is computed from the coefficients of variation of positive and negative peak amplitudes.
- Windows with \( Q \geq 0.7 \) are marked as valid.  
- A **joint validity mask** is formed as the logical AND of finger and wrist masks.

### APG-Based Diastolic Fallback

- When the diastolic peak is unclear (especially at the wrist), the second derivative PPG (APG) is used.
- The f‑point (valley after the e‑wave) in the APG serves as a proxy for the diastolic peak.
- This fallback is used to compute RI and the SI surrogate when direct morphology is insufficient.

---

## Headline Results (from Thesis)

These values are **from the original study** and are useful for sanity‑checking your own outputs:

| Metric | Wrist (error)                  | Finger (error)                 | Notes                                   |
|--------|--------------------------------|--------------------------------|-----------------------------------------|
| HR     | MAPE ≈ 1.22%, MAE ≈ 1.03 bpm  | MAPE ≈ 1.13%, MAE ≈ 0.97 bpm  | Well within 10% consumer threshold     |
| BR     | MAPE ≈ 13.33%, MAE ≈ 0.80 br/min | MAPE ≈ 6.50%, MAE ≈ 0.39 br/min | Wrist underestimates paced 6 br/min    |
| SpO₂   | MAE ≈ 7.5 percentage points    | MAE ≈ 2.1 percentage points    | Wrist outside ±2 pp clinical accuracy  |
| RI     | Wrist shows age effect but weaker separation vs finger | Finger shows clearer age-group separation | RI more robust than SI surrogate |

---

## Citation

If you use this code in academic work, please cite:

> M. S. Charan,  
> *“Comparative Assessment of Wrist and Finger Photoplethysmography for Multi-Metric Cardiovascular Monitoring,”*  
> BEng thesis, University of Edinburgh, 2026.

---


## Ethics

Data collection was approved under the University of Edinburgh School of Engineering Research Ethics framework. All participants provided written informed consent. No raw participant data is distributed with this repository in order to comply with GDPR and the UK Data Protection Act 2018. Consent form provided in consentform.pdf file.

---
