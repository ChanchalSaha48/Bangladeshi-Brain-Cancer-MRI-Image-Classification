# Brain Caner Classification from MRI using Deep Learning (VGG19 / VGG16 / ResNet50)

This repository contains a complete deep learning pipeline for **Brain Cancer Classification** from **MRI images** using multiple pretrained CNN architectures (VGG16, VGG19, ResNet50).  
The project includes **5-Fold Cross-Validation**, **Grad-CAM Explainability**, and clean modular code following research-grade structure.

---

## ⚙️ Requirements

- Python 3.9+  
- TensorFlow 2.x  
- Keras  
- OpenCV (`cv2`)  
- matplotlib, seaborn, pandas, scikit-learn  

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

Dataset Source: [**PMRAM: Bangladeshi Brain Cancer - MRI Dataset**](https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset) 
```md
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


** 📝 VGG16 Fold-wise Performance Metrics**

| Fold | Loss     | Accuracy | Precision | Recall   |
|------|----------|---------|-----------|-----------|
| 1    | 0.096516 | 0.965862 | 0.965862  | 0.965862 |
| 2    | 0.033066 | 0.987510 | 0.987510  | 0.987510 |
| 3    | 0.035797 | 0.987510 | 0.989158  | 0.987510 |
| 4    | 0.035942 | 0.986678 | 0.987500  | 0.986678 |
| 5    | 0.021066 | 0.990833 | 0.990833  | 0.990833 |

**Mean ± Std Dev:**
- Loss: 0.0445 ± 0.030  
- Accuracy: 0.9837 ± 0.010  
- Precision: 0.9842 ± 0.010  
- Recall: 0.9837 ± 0.010

---

** 📝 VGG19 Fold-wise Performance Metrics**

| Fold | Loss     | Accuracy | Precision | Recall   |
|------|----------|---------|-----------|---------|
| 1    | 0.134631 | 0.966777 | 0.966667 | 0.963455 |
| 2    | 0.155731 | 0.960133 | 0.963087 | 0.953488 |
| 3    | 0.286648 | 0.936877 | 0.939799 | 0.933555 |
| 4    | 0.151391 | 0.953488 | 0.956229 | 0.943522 |
| 5    | 0.164394 | 0.953488 | 0.956522 | 0.950166 |

**Mean ± Std Dev:**
- Loss: 0.1786 ± 0.0614  
- Accuracy: 0.9542 ± 0.0111  
- Precision: 0.9565 ± 0.0103  
- Recall: 0.9488 ± 0.0112

---

** 📝 ResNet50 Fold-wise Performance Metrics**

| Fold | Loss     | Accuracy | Precision | Recall   |
|------|----------|---------|-----------|---------|
| 1    | 0.652354 | 0.711907 | 0.771707 | 0.658618 |
| 2    | 0.680941 | 0.707744 | 0.762695 | 0.650291 |
| 3    | 0.741511 | 0.688593 | 0.754852 | 0.615321 |
| 4    | 0.758566 | 0.689425 | 0.756329 | 0.597003 |
| 5    | 0.700282 | 0.722500 | 0.789157 | 0.655000 |

**Mean ± Std Dev:**
- Loss: 0.7067 ± 0.0435  
- Accuracy: 0.7040 ± 0.0147  
- Precision: 0.7669 ± 0.0141  
- Recall: 0.6352 ± 0.0275

---

## Model Comparison Table (Summary)

| Model    | Loss ± SD      | Accuracy ± SD  | Precision ± SD | Recall ± SD    |
| -------- | -------------- | -------------- | -------------- | -------------- |
| VGG16    | 0.0441 ± 0.031 | 0.9837 ± 0.010 | 0.9848 ± 0.011 | 0.9837 ± 0.010 |
| VGG19    | 0.1786 ± 0.061 | 0.9542 ± 0.011 | 0.9565 ± 0.010 | 0.9488 ± 0.011 |
| ResNet50 | 0.7067 ± 0.043 | 0.7040 ± 0.015 | 0.7669 ± 0.014 | 0.6352 ± 0.027 |


### Grad-CAM Visualizations
See Grad-CAM overlays for VGG16, VGG19, and ResNet50 in `outputs/gradcam/`.

---
## 🏁 Conclusion

Based on 5-Fold Cross-Validation metrics, VGG16 achieved the highest accuracy (0.9837 ± 0.010) and precision (0.9848 ± 0.011), followed by VGG19 and ResNet50.  
Grad-CAM visualizations confirm that VGG16 focuses correctly on tumor regions, providing reliable interpretability.  
Therefore, VGG16 with fine-tuning is the preferred model for multi-class brain tumor MRI classification in this dataset.

---

📚 Citation :

If you use this repository, please cite:@article{chanchalsaha2025braintumor,
-  title={MRI Brain Tumor Classification using Deep Transfer Learning and 5-Fold Cross-Validation},
-  author={Chanchal Saha},`
-  year={2025}
}


📬 Contact

Author: Chanchal Saha
Email: chanchalsaha415@gmail.com
