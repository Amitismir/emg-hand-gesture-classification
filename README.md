# EMG Hand Gesture Classification

Machine learning project for classification of hand and wrist movements using surface EMG signals.

## 📊 Project Overview

This project processes multi-channel EMG signals and classifies hand gestures using signal processing and machine learning techniques.

Dataset contains:

* 4 EMG channels
* 21 hand movements
* 10 repetitions per movement
* Recorded from forearm muscles

## ⚙️ Pipeline

1. Load and parse `.mat` EMG dataset
2. Signal preprocessing & segmentation
3. Feature extraction (MAV, RMS, WL, ZC, SSC, VAR, etc.)
4. Machine learning models:

   * KNN
   * SVM
   * Random Forest
   * LDA
5. Evaluation using accuracy & confusion matrix

## 📁 Project Structure

```
data/        → raw EMG data (not uploaded)
src/         → source code
notebooks/   → experiments
results/     → outputs & figures
```

## 🚀 How to Run

Install dependencies:

```
pip install numpy scipy pandas matplotlib scikit-learn seaborn
```

Run main script:

```
python main.py
```

## 👨‍💻 Author

Amitis Mirabedini
Sharif University of Technology
Sensing and Measurement course project

