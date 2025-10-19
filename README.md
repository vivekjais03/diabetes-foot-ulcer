# 🦶 Foot Ulcer Detection & Analysis System

An AI-powered web application for early detection of foot ulcers using deep learning and computer vision. The system provides comprehensive analysis, Grad-CAM visualizations, and personalized care recommendations.

## ✨ Features

### 🎯 Core Functionality
- **Image Upload**: Drag & drop or file selection interface
- **AI Detection**: Binary classification (Normal vs Abnormal/Ulcer)
- **Confidence Scoring**: Detailed confidence metrics with threshold control
- **Grad-CAM Visualization**: Heatmap overlay showing where the AI focused
- **PDF Reports**: Comprehensive medical reports with findings and recommendations

### 🚀 Advanced Features
- **Confidence Threshold Control**: Adjustable sensitivity (50%-95%)
- **Severity Assessment**: Automatic severity classification based on confidence
- **Care Recommendations**: Personalized remedies and care tips
- **Interactive Chatbot**: AI assistant for ulcer care questions
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### 🔬 Technical Features
- **Convolutional Neural Network (CNN)**: Deep learning model for image classification
- **Grad-CAM**: Explainable AI showing decision-making process
- **Real-time Processing**: Fast image analysis and results display
- **Secure File Handling**: Safe image upload and processing

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (for model inference)
- Web browser with JavaScript enabled

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd foot_ulcer_project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Model
Ensure you have a trained model file:
```bash
# If you don't have a model, train one first:
python notebooks/train_model.py
```

### Step 4: Run Application
```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## 📁 Project Structure

```
foot_ulcer_project/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── models/                         # Trained model files
│   └── foot_ulcer_model.h5        # Your trained model
├── notebooks/                      # Jupyter notebooks and scripts
│   ├── enhanced_model.py          # Enhanced model with Grad-CAM
│   ├── train_model.py             # Model training script
│   ├── test_model.py              # Model testing script
│   └── preprocess_dataset.py      # Dataset preprocessing
├── templates/                      # HTML templates
│   └── index.html                 # Main web interface
├── dataset/                        # Dataset directory
│   └── split_dataset/             # Train/validation/test splits
├── uploads/                        # Uploaded images and generated reports
└── results/                        # Analysis results and outputs
```

## 🚀 Usage Guide

### 1. Web Interface
1. **Open Application**: Navigate to `http://localhost:5000`
2. **Upload Image**: Drag & drop or click to select a foot image
3. **Set Threshold**: Adjust confidence threshold using the slider
4. **Analyze**: Click upload to process the image
5. **View Results**: See prediction, confidence, and Grad-CAM heatmap
6. **Generate Report**: Download comprehensive PDF report
7. **Get Care Tips**: View personalized recommendations

### 2. Confidence Thresholds
- **50-60%**: High sensitivity, more false positives
- **70-80%**: Balanced sensitivity and specificity (recommended)
- **85-95%**: High specificity, fewer false positives

### 3. Understanding Results
- **Normal (Healthy skin)**: Green border, healthy status
- **Abnormal (Ulcer)**: Red border, severity level indicated
- **Confidence Score**: Percentage indicating prediction certainty
- **Grad-CAM Heatmap**: Red areas show where AI detected abnormalities

### 4. Chatbot Usage
Ask questions about:
- "How to clean an ulcer?"
- "Signs of infection?"
- "How to prevent ulcers?"
- "Proper dressing techniques?"
- "Healing promotion tips?"

## 🔬 Model Architecture

### CNN Structure
```
Input: 128x128x3 RGB images
├── Conv2D(32, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(64, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(128, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Flatten()
├── Dense(128) + ReLU + Dropout(0.5)
└── Output: Dense(1) + Sigmoid
```

### Training Details
- **Dataset**: Binary classification (Normal vs Abnormal)
- **Augmentation**: Rotation, shift, zoom, flip
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy

## 📊 Dataset Information

### Current Dataset
- **Normal Images**: 540 healthy foot images
- **Abnormal Images**: 500 ulcer images
- **Split**: Train (70%) / Validation (15%) / Test (15%)

### Dataset Structure
```
dataset/
├── Patches/
│   ├── Normal(Healthy skin)/     # 540 images
│   └── Abnormal(Ulcer)/         # 500 images
└── split_dataset/
    ├── train/                    # Training data
    ├── val/                      # Validation data
    └── test/                     # Test data
```

## 🔧 Customization

### Adding New Classes
To extend to multi-class classification (e.g., ulcer stages):

1. **Modify Model Architecture**:
```python
# Change output layer
Dense(num_classes, activation='softmax')  # Instead of sigmoid
```

2. **Update Training Script**:
```python
# Change class mode
class_mode='categorical'  # Instead of 'binary'
```

3. **Modify Enhanced Model**:
```python
# Update prediction logic for multiple classes
predictions = model.predict(img_array)
class_index = np.argmax(predictions[0])
```

### Custom Remedies
Edit the `get_remedies()` method in `enhanced_model.py` to add:
- New severity levels
- Custom care recommendations
- Language-specific content
- Specialized medical advice

## 🚨 Important Notes

### Medical Disclaimer
- **This tool is for screening purposes only**
- **Always consult healthcare professionals for medical decisions**
- **Not a replacement for professional medical diagnosis**
- **Use as part of comprehensive foot care routine**

### Performance Considerations
- **Model accuracy depends on training data quality**
- **Image quality affects prediction reliability**
- **Regular model retraining recommended with new data**
- **Consider clinical validation for medical use**

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   Error: Model not found at models/foot_ulcer_model.h5
   Solution: Train the model first using train_model.py
   ```

2. **Memory Issues**
   ```
   Error: CUDA out of memory
   Solution: Reduce batch size or use CPU-only mode
   ```

3. **Import Errors**
   ```
   Error: Module not found
   Solution: Install requirements: pip install -r requirements.txt
   ```

4. **Image Upload Fails**
   ```
   Error: File upload failed
   Solution: Check file size (max 16MB) and format (JPG/PNG)
   ```

### Performance Tips
- Use GPU acceleration if available
- Optimize image size (128x128 recommended)
- Close other applications to free memory
- Use SSD storage for faster file operations

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: Hindi, Spanish, French
- **Mobile App**: React Native or Flutter
- **Cloud Deployment**: AWS/Azure integration
- **Real-time Video**: Live foot inspection
- **Integration**: EHR system connectivity

### Research Areas
- **Stage Classification**: Early, moderate, severe ulcer detection
- **Infection Detection**: Bacterial vs fungal classification
- **Healing Progress**: Longitudinal analysis over time
- **Risk Assessment**: Predictive analytics for ulcer development

## 🤝 Contributing
1. Nikhil Gupta
2. Aakriti Musaddi

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

### Code Standards
- Follow PEP 8 Python style guide
- Add docstrings for all functions
- Include type hints where possible
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

### Getting Help
- **Issues**: Create GitHub issue with detailed description
- **Documentation**: Check this README and code comments
- **Community**: Join our discussion forum

### Contact Information
- **Project Maintainer**: Vivek Jaiswal
- **Email**: jaiswalvivek421@gmail.com
- **GitHub**: @vivekjais03

---

## 🎯 Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Clone repository
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Ensure model exists: `models/foot_ulcer_model.h5`
- [ ] Run application: `python app.py`
- [ ] Open browser: `http://localhost:5000`
- [ ] Upload test image
- [ ] Verify results display
- [ ] Test PDF generation
- [ ] Try chatbot functionality

**🎉 You're ready to detect foot ulcers with AI!**
