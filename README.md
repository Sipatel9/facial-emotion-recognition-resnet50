# Facial Emotion Recognition Using Deep Learning (ResNet50) 🤖

Deep learning-based facial emotion recognition using transfer learning with ResNet50 architecture on JAFFE and CK+ datasets.

## 📋 Overview

This project implements a state-of-the-art facial emotion recognition system using:
- **ResNet50** pre-trained model with transfer learning
- **JAFFE & CK+** emotion datasets
- **PyTorch/TensorFlow** for model training and inference
- Advanced data augmentation and preprocessing techniques
- Comprehensive evaluation metrics and visualizations

## 🎯 Features

✅ Transfer learning with ResNet50 backbone  
✅ Multi-dataset training (JAFFE, CK+)  
✅ Emotion classification (7-8 emotions)  
✅ Real-time prediction capabilities  
✅ Model evaluation with detailed metrics  
✅ Data visualization and analysis  
✅ Early stopping and learning rate scheduling  

## 🧬 Emotions Recognized

- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😐 Neutral
- 😲 Surprised
- 😨 Fear
- 🤮 Disgust

## 📊 Dataset Information

| Dataset | Images | Emotions | Resolution |
|---------|--------|----------|------------|
| JAFFE | 213 | 7 | 256×256 |
| CK+ | 593 | 8 | 640×490 |

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
pytorch >= 1.9 or tensorflow >= 2.6
torchvision >= 0.10
numpy, pandas, matplotlib, scikit-learn, opencv-python
```

### Installation
```bash
git clone https://github.com/Sipatel9/facial-emotion-recognition-resnet50.git
cd facial-emotion-recognition-resnet50
pip install -r requirements.txt
```

### Usage
```python
# Load and use the model
from model import EmotionRecognizer

recognizer = EmotionRecognizer(model_path='models/resnet50_emotion_model.pth')
emotions = recognizer.predict_from_image('image.jpg')
```

## 📈 Model Performance

### JAFFE Dataset
- **Accuracy**: 40.0%
- **Precision**: 0.5688
- **Recall**: 0.3944
- **F1-Score**: 0.4049

### CK+ Dataset
- **Accuracy**: 56.7%
- **Precision**: 0.5114
- **Recall**: 0.4927
- **F1-Score**: 0.4702

CK+ performs better due to larger size and more diverse subjects.

## 🏗️ Architecture

```
Input Image (224×224)
    ↓
ResNet50 Backbone (Pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Custom Dense Layers
    ↓
Dropout (0.5)
    ↓
Softmax Output (7-8 emotions)
```

## 📚 Key Techniques

- **Transfer Learning** - Leverage pre-trained ImageNet weights
- **Two-Stage Training** - Frozen base → fine-tuning
- **Data Augmentation** - Rotation, zoom, brightness adjustments
- **Batch Normalization** - Improved training stability
- **Dropout Regularization** - Prevent overfitting
- **Learning Rate Scheduling** - Adaptive learning rates
- **Class Weighting** - Handle class imbalance

## 📁 Project Structure

```
├── data/
│   ├── jaffe/
│   └── ck_plus/
├── figures/
│   ├── confusion_matrix_jaffe.png
│   ├── confusion_matrix_ckplus.png
│   ├── accuracy_loss_jaffe.png
│   ├── accuracy_loss_ckplus.png
│   ├── sample_predictions.png
│   └── accuracy_comparison.png
├── models/
│   └── resnet50_emotion_model.pth
├── notebooks/
│   └── emotion_recognition.ipynb
├── src/
│   ├── model.py
│   ├── train.py
│   └── predict.py
└── README.md
```

## 📈 Visual Results

All training results, confusion matrices, and performance charts are included in the `figures/` folder:

- **Confusion Matrices** for both datasets
- **Training/Validation Curves** showing accuracy and loss
- **Sample Predictions** on test images
- **Accuracy Comparison Charts**

## 🔗 Resources

- 📘 [Full Academic Report](report/)
- 🔗 [Google Colab Notebook](https://colab.research.google.com/drive/1E2Zs1vyOLt6oo-rO3VV7oyalcLREoJBv?usp=sharing)
- 📊 [JAFFE Dataset](https://www.kasrl.org/jaffe.html)
- 📊 [CK+ Dataset](https://www.jeffcohn.com/databases/)

## 🎓 Key Insights

- Transfer learning significantly improves FER performance on small datasets
- Larger, more diverse datasets (CK+) lead to better generalization
- Two-stage training approach balances accuracy and computational efficiency
- Attention mechanisms and ensemble methods could further improve results

## 🚀 Future Improvements

- Add attention mechanisms (CBAM, SE blocks)
- Use larger datasets (FER2013, RAF-DB)
- Implement ensemble models
- Improve generalization with face alignment
- Deploy as web/mobile application

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## 👩‍💻 Author

**Samira Patel**  
BSc (Hons) Computer Science  
University of Central Lancashire (UCLan)

## 📞 Contact

For questions or collaborations: [GitHub Profile](https://github.com/Sipatel9)

---

**⭐ If you found this helpful, please consider giving it a star!**
