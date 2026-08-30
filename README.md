<div align="center">

# ✈️ Pilot Fatigue Risk Detection System
### *Intelligent Aviation Safety & Physiological Predictive Modeling*

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PhysioNet WFDB](https://img.shields.io/badge/PhysioNet-WFDB%20Signals-006699?style=for-the-badge)](https://physionet.org)
[![Status](https://img.shields.io/badge/Status-Completed%20%26%20Benchmarked-success?style=for-the-badge)](#results)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Executive Summary

Fatigue in aviation operations is a critical human-factor risk factor directly linked to cognitive degradation, reduced situational awareness, and delayed reaction times. 

The **Pilot Fatigue Risk Detection System** is an end-to-end machine learning pipeline that fuses **real-time physiological signal analytics (ECG/HRV)** with **operational human factor parameters** (sleep debt, duty cycle, circadian disruption, flight hours) to compute an objective **Fatigue Risk Score (FRS)** and classify impairment levels before safety margins are compromised.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Acquisition ["1. Data Acquisition & Signals"]
        A1[Physiological ECG / HRV Signals] --> B1[PhysioNet WFDB Loader]
        A2[Operational Factors: Sleep Debt, Duty Hours, Circadian Index] --> B2[Human Factors Feature Vector]
    end

    subgraph Preprocessing ["2. Feature Engineering & Preprocessing"]
        B1 --> C1[Signal Windowing & Filtering]
        C1 --> C2[Time-Domain HRV Extraction\n(SDNN, RMSSD, pNN50)]
        B2 & C2 --> D1[Feature Aggregator & MinMaxScaler]
        D1 --> D2[SelectKBest Feature Optimization]
    end

    subgraph ML_Engine ["3. Machine Learning Inference Engine"]
        D2 --> E1{Trained Classifier Ensemble}
        E1 -->|Benchmark #1| F1[Random Forest]
        E1 -->|Top Model (81.56%)| F2[Support Vector Machine - SVM]
        E1 -->|Benchmark #2| F3[K-Nearest Neighbors - KNN]
    end

    subgraph Output_Layer ["4. Risk Assessment & Decision Support"]
        F2 --> G1[Fatigue Risk Score: FRS Engine]
        G1 --> H1[Low Risk: Operational Go]
        G1 --> H2[Moderate Risk: Advisory Alert]
        G1 --> H3[High Risk: Fatigue Intervention Required]
    end
```

---

## 🧠 Key Features & Innovations

- 📈 **Physiological HRV Feature Extraction**: Extracts time-domain heart rate variability metrics ($SDNN$ and $RMSSD$) indicating autonomic nervous system stress and alertness degradation.
- 🕒 **Circadian & Duty Cycle Modeling**: Factors in time-of-day circadian rhythm troughs and multi-leg cumulative flight hours.
- ⚖️ **Multi-Model Benchmark Matrix**: Systematically compares non-linear kernels (RBF-SVM), ensemble decision trees (Random Forest), and distance-based heuristics (KNN).
- 🛡️ **Defensive Error Handling & Data Quality Checks**: Automated null interpolation, outlier clipping, and standardized MinMax feature scaling.

---

## 📊 Benchmark & Performance Results

Models were evaluated using stratified k-fold validation on synchronized physiological and operational datasets:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF Kernel)** | **81.56%** | **0.82** | **0.81** | **0.81** | 🏆 **Optimal Performer** |
| **Random Forest Classifier (100 Trees)** | 80.59% | 0.80 | 0.81 | 0.80 | Robust Ensemble |
| **K-Nearest Neighbors ($k=5$)** | 78.92% | 0.78 | 0.79 | 0.78 | Baseline Heuristic |

> [!TIP]
> The **RBF-kernel Support Vector Machine** delivered the highest generalization capability across boundary-case fatigue transitions with minimal false negatives.

---

## 📂 Repository Structure

```
pilot-fatigue-risk-detection/
├── data/                       # Processed datasets and sample physiological records
├── models/                     # Serialized model weights & scalers (.joblib / .pkl)
├── src/
│   ├── data_loader.py          # PhysioNet signal ingestion & feature parsing
│   ├── feature_extraction.py   # HRV calculation (SDNN, RMSSD) & human-factor fusion
│   ├── train.py                # Hyperparameter tuning and model training
│   └── evaluate.py             # Metrics calculation (ROC-AUC, Confusion Matrix)
├── main.py                     # Primary execution pipeline & interactive prediction
├── requirements.txt            # Dependency specifications
└── README.md                   # Technical documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AyushrathoreCs264/Pilot-Fatigue-Detection.git
cd Pilot-Fatigue-Detection
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the Pipeline
```bash
python main.py
```

---

## 🗺️ Roadmap & Future Enhancements

- [ ] **Edge Deployment**: Integration with lightweight Raspberry Pi / Jetson Nano on-cockpit telemetry.
- [ ] **Wearable Sensor Integration**: Live streaming ingestion via BLE from smartwatches (PPG/SpO2).
- [ ] **Deep Learning Integration**: Temporal 1D-CNN + LSTM networks for continuous raw signal stream sequence modeling.

---

## 👥 Authors & Acknowledgments

- **Ayush Rathore** ([@AyushrathoreCs264](https://github.com/AyushrathoreCs264)) — *System Architecture, ML Pipeline Design, Model Benchmarking & Risk Scoring Algorithms*
- **Mahi** — *Physiological Signal Acquisition & HRV Feature Engineering*
- **Prachi** — *Data Preprocessing, Normalization & Evaluation Documentation*

---

<div align="center">
  <sub>Developed for academic research in Human Factors and Aviation Safety Automation.</sub>
</div>
