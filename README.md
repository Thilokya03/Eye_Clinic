# Eye Clinic - Automated Glaucoma Screening System

## Overview

**Automated Glaucoma Screening Using Optic Disc-Cup Segmentation and Hybrid Machine Learning Model**

This project implements a comprehensive machine learning pipeline for automated glaucoma detection using fundus images. It combines deep learning for image segmentation with classical machine learning classifiers to achieve robust glaucoma screening.

## 🎯 Project Objectives

- Segment optic disc and cup regions from fundus images using U-Net architecture
- Extract morphological features from segmented regions
- Classify glaucoma using a hybrid approach combining CNN features and XGBoost classifier
- Provide a scalable solution for eye clinic screening workflows

## 📊 Dataset

The project utilizes multiple glaucoma datasets:
- **G1020** - Large-scale glaucoma dataset
- **ORIGA** - Optic Retinal Images for Glaucoma Analysis
- **REFUGE** - Retinal Fundus Glaucoma Detection Challenge

All images are standardized to **512×512 resolution** for consistent processing.

## 🏗️ Architecture

### 1. Image Segmentation
- **Model**: U-Net with ResNet-34 backbone
- **Task**: Optic disc and cup segmentation
- **Output**: Segmentation masks for morphological analysis

### 2. Feature Extraction
- CNN-based feature extraction from original images
- Morphological features from segmentation masks
- Hybrid feature set combining visual and geometric characteristics

### 3. Classification
- **Primary Classifier**: XGBoost with calibration
- **Calibration**: CalibratedClassifierCV for probability estimation
- **Task**: Binary classification (Glaucoma / Non-Glaucoma)

## 📁 Project Structure

```
Eye_Clinic/
├── Models/
│   ├── best_unet_resnet34.pth           # U-Net segmentation model
│   ├── best_image_classifier.pth        # CNN feature extractor
│   ├── xgb_glaucoma_uncalibrated.json   # XGBoost classifier
│   ├── model4_xgb_calibrated.joblib     # Calibrated XGBoost
│   └── model4_meta_info.joblib          # Model metadata
├── Docs/
│   └── Eye_Clinic.ipynb                 # Complete project notebook
├── Models/train_df.csv                  # Training dataset metadata
├── Models/test_df.csv                   # Testing dataset metadata
└── README.md                            # This file
```

## 🚀 Key Features

✅ **End-to-End Pipeline**: From image preprocessing to glaucoma prediction
✅ **Multiple Datasets**: Trained on G1020, ORIGA, and REFUGE datasets
✅ **Robust Segmentation**: U-Net architecture for precise optic disc-cup delineation
✅ **Calibrated Predictions**: Probabilistic outputs suitable for clinical decision-making
✅ **Feature Engineering**: Combines deep learning and morphological features
✅ **Production-Ready Models**: Pre-trained weights available for inference

## 💻 Usage

For complete implementation details, training procedures, and inference examples, refer to the [full notebook](https://colab.research.google.com/drive/1kVfss0_he07A4yGU4ywETMOHBdcBFrLP?usp=sharing).

## 🔧 Model Details

| Model | Purpose | Framework |
|-------|---------|-----------|
| best_unet_resnet34.pth | Optic disc-cup segmentation | PyTorch |
| best_image_classifier.pth | CNN feature extraction | PyTorch |
| model4_xgb_calibrated.joblib | Glaucoma classification | XGBoost |

## 📈 Performance

The hybrid approach leverages:
- **Deep Learning**: State-of-the-art segmentation and feature extraction
- **Classical ML**: Robust classification with interpretable decision boundaries
- **Calibration**: Reliable probability estimates for clinical use

## 🔗 References & Resources

- [Complete Colab Notebook](https://colab.research.google.com/drive/1kVfss0_he07A4yGU4ywETMOHBdcBFrLP?usp=sharing)
- Glaucoma Detection Challenge (REFUGE)
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- XGBoost: A Scalable Tree Boosting System

## 📝 Documentation

- **Full Report**: See [Eye - Clinic_Report.pdf](Docs/Eye%20-%20Clinic_Report.pdf) in the Docs folder for detailed analysis and results

## 👥 Team

This is a collaborative team project by:

[Thilokya Angeesa](https://github.com/Thilokya03)
[Nadil Kulathunga](https://github.com/nadilHesara)
[Chamodh Nethsara](https://github.com/chamodhk) 

## 📝 License

See LICENSE file for details.

---

Developed as an automated screening solution for eye clinics and ophthalmic research.