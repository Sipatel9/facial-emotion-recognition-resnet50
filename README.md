# Facial Emotion Recognition Using Deep Learning (ResNet50)

This project implements a facial emotion recognition system using **ResNet50 with transfer learning**, evaluated on the **JAFFE** and **CK+** datasets. The model classifies seven basic emotions and compares performance across both datasets using standard deep‑learning evaluation metrics.

---

## 📌 Project Overview

- Built a deep‑learning FER model using **ResNet50 pre‑trained on ImageNet**
- Evaluated on **JAFFE** (213 images) and **CK+** (593 images)
- Implemented:
  - Image preprocessing (RGB conversion, resizing to 224×224)
  - Data augmentation
  - Class weighting for imbalance
  - Two‑stage training (frozen base → fine‑tuning)
- Compared performance using:
  - Accuracy
  - Precision
  - Recall
  - F1‑score
  - Confusion matrices
  - Training/validation curves

This project demonstrates how transfer learning improves FER performance on small datasets compared to traditional ML methods (HOG/LBP + SVM).

---

## 🧠 Deep Learning Approach

### **Model Architecture**
- Base model: **ResNet50**
- Pretrained on: **ImageNet**
- Custom layers added:
  - GlobalAveragePooling
  - Dense layer
  - Dropout
  - Final softmax classifier

### **Training Strategy**
1. **Stage 1:** Freeze ResNet50 base → train classifier only  
2. **Stage 2:** Unfreeze top layers → fine‑tune with low learning rate  

### **Loss & Optimizer**
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- Callbacks: EarlyStopping, ReduceLROnPlateau

---

## 📊 Results Summary

### **JAFFE**
- Accuracy: **40.0%**
- Precision: **0.5688**
- Recall: **0.3944**
- F1‑score: **0.4049**

### **CK+**
- Accuracy: **56.7%**
- Precision: **0.5114**
- Recall: **0.4927**
- F1‑score: **0.4702**

CK+ performs better due to larger size and more diverse subjects.

---

## 📂 Repository Structure

---

## 📈 Visual Outputs

## 📂 Figures

All visual outputs used in this project are stored in the `figures/` folder, including:

- Confusion Matrix (JAFFE)
- Confusion Matrix (CK+)
- Training & Validation Accuracy/Loss Curves (JAFFE)
- Training & Validation Accuracy/Loss Curves (CK+)
- Sample Emotion Predictions
- Accuracy Comparison Chart
- Additional visualisations from the analysis

Below are previews of the figures included in the repository:

![Confusion Matrix JAFFE](figures/confusion_matrix_jaffe.png)
![Accuracy Loss JAFFE](figures/accuracy_loss_jaffe.png)
![Accuracy Loss CK+](figures/accuracy_loss_ckplus.png)
![Confusion Matrix CK+](figures/confusion_matrix_ckplus.png)
![Sample Predictions](figures/sample_predictions.png)
![Accuracy Comparison](figures/accuracy_comparison.png)

---

## 📘 Full Report

The full academic report is included in the **report/** folder.[Facial Emotion recognition Using Deep Learning.docx](https://github.com/user-attachments/files/27244619/Facial.Emotion.recognition.Using.Deep.Learning.docx)


---

## 🔗 Google Colab Notebook

Add your Colab link here:
https://colab.research.google.com/drive/1E2Zs1vyOLt6oo-rO3VV7oyalcLREoJBv?usp=sharing



---

## 🚀 Future Improvements

- Add attention mechanisms (CBAM, SE blocks)  
- Use larger datasets (FER2013, RAF‑DB)  
- Implement ensemble models  
- Improve generalisation with face alignment  

---

## 👩‍💻 Author

**Samira Patel**  
BSc (Hons) Computer Science  
University of Central Lancashire (UCLan)


