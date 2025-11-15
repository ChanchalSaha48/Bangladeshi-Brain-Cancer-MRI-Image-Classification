# Brain Caner Classification from MRI using Deep Learning (VGG19 / VGG16 / ResNet50)

This repository contains a complete deep learning pipeline for **Brain Cancer Classification** from **MRI images** using multiple pretrained CNN architectures (VGG19, VGG16, ResNet50).  
The project includes **5-Fold Cross-Validation**, **Grad-CAM Explainability**, and clean modular code following research-grade structure.

---

## 🚀 Key Features

- ✔️ MRI-based Brain Cancer Classification  
- ✔️ Transfer Learning with:  
  - VGG19  
  - VGG16  
  - ResNet50  
- ✔️ 5-Fold Cross-Validation  
- ✔️ Grad-CAM Visualization for interpretability  
- ✔️ High-quality modular code (dataset, model, training, utils)  
- ✔️ GPU/CPU both supported  
- ✔️ Easy to reproduce results  

---

## 🧠 Dataset

- **Modality:** MRI  
- **Task:** Brain Cancer Classification  
- **Classes:** e.g., Glioma, Meningioma, Pituitary, No-Tumor  

Add your dataset source here:

```md
Dataset Source: https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset 

dataset/
│
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```
---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Training (5-Fold Cross-Validation)

- Each notebook trains one model (VGG19 / VGG16 / ResNet50)  
- **Procedure:**
  1. Split dataset into 5 folds  
  2. Train model on 4 folds and validate on 1 fold  
  3. Repeat for all folds  
  4. Fine-tune last 30% of convolutional layers  
  5. Evaluate fold-wise metrics: Loss, Accuracy, Precision, Recall  
  6. Save best model per fold (`outputs/best_model_foldX.h5`)  

- Metrics summary saved as DataFrame: `Result`  
- Grad-CAM visualizations saved in: `outputs/gradcam/`  

---

## 🔬 Grad-CAM (Interpretability)

- Used to visualize model attention regions on MRI images  
- Overlay of Grad-CAM heatmap on original MRI  
- True and predicted labels displayed as title  
- Paper-ready high-resolution images (`dpi=300`)  

---

## 📊 Results


# For VGG16
- Fold-wise metrics table: `Loss | Accuracy | Precision | Recall`

- Mean ± Std deviation across 5 folds  
- Grad-CAM visualizations for selected images  

---

## ⚙️ Requirements

- Python 3.9+  
- TensorFlow 2.x  
- Keras  
- OpenCV (`cv2`)  
- matplotlib, seaborn, pandas, scikit-learn  

```bash
pip install -r requirements.txt




