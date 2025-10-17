# 🎗️ Breast Cancer Prediction App

Predict whether a tumor is **malignant** or **benign** using a pre-trained machine learning model. This project uses selected features from the **Wisconsin Breast Cancer Dataset** and a **Random Forest Classifier** to provide accurate predictions.

---

## 🧠 Features & Inputs

The model uses **5 key tumor features** (out of the 30 available in the dataset) to simplify user input:

| Feature             | Description                                               |
|--------------------|-----------------------------------------------------------|
| **Mean Radius**     | Average distance from the center to points on the tumor boundary |
| **Mean Texture**    | Standard deviation of gray-scale values in the tumor image |
| **Mean Perimeter**  | Average perimeter of the tumor                            |
| **Mean Area**       | Average area of the tumor                                 |
| **Mean Smoothness** | Variation in radius lengths (how smooth the tumor boundary is) |

These features are collected via **sliders in the Streamlit sidebar**, allowing interactive input.

---

## ⚙️ How the Model Works

1. **Dataset**: Uses the Wisconsin Breast Cancer Dataset (`sklearn.datasets.load_breast_cancer`).  
2. **Feature Selection**: Only 5 features are selected for simplicity and quick deployment.  
3. **Model**: Trained a **Random Forest Classifier** using these features.  
   - Random Forests are ensemble models that combine multiple decision trees to improve prediction accuracy.  
   - They handle non-linear relationships and reduce overfitting.  
4. **Prediction**: The model outputs:
   - **Label**: `Benign 🟢` or `Malignant 🔴`  
   - **Probability**: Shows the likelihood of each class.  

---

## 🏃 Steps to Run Locally

1. **Clone the Repository** (or download the files):
```bash
git clone git@github.com:Gleek231997/Breast-Cancer-Prediction.git
cd Breast-Cancer-Prediction
```
2. **Install Dependencies**:
```bash
pip install -r requirments.txt
```
3.**Run the App**:
```bash
streamlit run app.py
```
4.**Ineract**:
Use the sidebar sliders to input the 5 tumor features.
Click Predict to see the prediction and probability.

## 📊 Output Example
**Bengin Prediction**
<img width="1436" height="954" alt="Screenshot 2025-10-17 at 1 51 43 PM" src="https://github.com/user-attachments/assets/9989225a-a7fa-4b66-9b8d-2e2ce274d4a2" />


**Malignant Prediction**
<img width="1436" height="954" alt="Screenshot 2025-10-17 at 1 51 53 PM" src="https://github.com/user-attachments/assets/53454bfb-ee1e-43fa-93d3-49baa88a4ecb" />


**🔧 Tech Stack**
```bash
Python 3.x
Streamlit – for interactive UI
Scikit-learn – for ML model
NumPy & Pandas – data handling
```


