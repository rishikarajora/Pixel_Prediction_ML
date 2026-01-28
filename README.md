# Pixel Coordinate Prediction using Deep Learning

## Problem Statement
Predict the (x, y) coordinates of a single active pixel (value = 255) in a 50×50 grayscale image where all other pixels are 0.

---

## Approach
This problem is treated as a **coordinate regression task** using a Convolutional Neural Network (CNN).

- The input space is finite (50×50 grid), so a **deterministic dataset** is generated using a **2D meshgrid**.  
- Each point in the meshgrid corresponds to one possible pixel location: `(x, y)`.  
- For each location, an image is created with **exactly one active pixel**, and the label is its **normalized (x, y) coordinates**.  
- This ensures **full coverage of the input space**, avoids duplicate samples, and provides a **reproducible, production-ready dataset**.

---

## Dataset Details
- **Total samples:** 2500 (one for each pixel in a 50×50 grid)  
- **Image size:** 50×50 (grayscale)  
- **Active pixel value:** 1.0 (normalized from 255)  
- **Label format:** Normalized (x, y) coordinates  
- **Generation method:** 2D meshgrid of all pixel positions

---

## Model Architecture
- Convolutional layers for spatial feature extraction  
- MaxPooling for dimensionality reduction  
- Fully connected layers for coordinate regression  
- Sigmoid activation for normalized output  

**Loss Function:** Mean Squared Error (MSE)  
**Evaluation Metric:** Mean Absolute Error (MAE)  

---

## Training Details
- **Train-validation split:** 80% / 20%  
- **Optimizer:** Adam  
- **Epochs:** 10  
- **Batch size:** 32  

---

## Results
- **Final Validation MSE:** ~0.00088  
- **Final Validation MAE:** ~0.02  
- Approx. **1-pixel localization error**, acceptable for this task.

---

## Meshgrid Dataset Generation
The dataset is generated using a **2D meshgrid approach**:

```python
import numpy as np

img_size = 50
images = []
labels = []

for y in range(img_size):
    for x in range(img_size):
        img = np.zeros((img_size, img_size), dtype=np.float32)
        img[y, x] = 1.0  # Active pixel
        images.append(img)
        labels.append([x/img_size, y/img_size])  # Normalized coordinates

images = np.array(images).reshape(-1, img_size, img_size, 1)
labels = np.array(labels, dtype=np.float32)

print("Dataset shape:", images.shape, labels.shape).

---

```

## Run Instructions
### 1. Clone Repository
```bash
git clone https://github.com/rishikarajora/Pixel_Prediction_ML
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run Notebook
```bash
Open pixel_coordinate_prediction.ipynb in Jupyter Notebook
```
### 4. Run Python Script
```bash
python model_training.py
```

