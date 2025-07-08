# 🩺 ECG Analysis Web Application

## 🔬 Overview
This web-based **ECG Analysis Application** uses **both Deep Learning and Machine Learning with Image Processing techniques** to detect cardiovascular abnormalities from ECG images. It enables healthcare professionals to **upload ECG scans**, **enter patient information**, and receive predictions from **two AI models** — one powered by Deep Learning Roboflow and another by traditional Machine Learning + Advance Image Processing (with PCA and classification model).

---

## 🚀 Features

- ✅ **Login Authentication** for secure access  
- ✅ **ECG Image Upload** via Web UI  
- ✅ **Patient Metadata Collection** (Name, Age, Gender, Doctor Name, Symptoms)  
- ✅ **Dual AI Prediction** using:  
  - 🧠 **Deep Learning (YOLO via Roboflow)**  
  - ⚙️ **Machine Learning (Advance Image Processing + PCA + Classifier)**  
- ✅ **Processed Image Visualization** (Grayscale, Preprocessed Leads, Contours)  
- ✅ **1D Signal + Dimensionality Reduction (PCA)** display  
- ✅ **Downloadable Output Images**  
- ✅ **Modern Responsive UI** with **Bootstrap**  
- ✅ **Deployment Ready** on **Render with Gunicorn**  

---

## 🛠 Tech Stack

| Category       | Tech Used                          |
|----------------|------------------------------------|
| **Frontend**   | HTML, CSS, Bootstrap               |
| **Backend**    | Flask, OpenCV, NumPy, Pandas       |
| **Deep Learning** | YOLOv10, Supervision              |
| **ML Model**   | PCA (Principal Component Analysis), Pickle Classifier, Random Forest, Encoder |
| **Deployment** | Gunicorn, Render                   |
| **Python Version** | `Python 3.7.0`                    |

---

## 📂 Folder Structure

```
/ecg-analysis-app
│
├── app.py                     # Main Flask App
├── Ecg.py                     # ECG Preprocessing & ML Prediction Module
├── requirements.txt           # Dependencies
├── README.md
├── Procfile                   # Render Deployment File
│
├── model/
│   ├── PCA_ECG.pkl
│   └── Heart_Disease_Predictor.pkl
│
├── process/                   # Machine Learning generated files
│   ├── gray_image.png
│   ├── prediction1.jpg
│   ├── prediction2.jpg
│   ├── Contour_Leads_1-12_figure.png
│   ├── Preprocessed_Leads_1-12_figure.png
│   ├── Preprocessed_Leads_13_figure.png
│   ├── Leads_1-12_figure.png
│   ├── Long_Lead_13_figure.png
│   ├── 1d_signal.csv
│   ├── reduced_signal.csv
│   ├── Scaled_1DLead_1.csv ... Scaled_1DLead_12.csv
│
├── uploads/                   # User-uploaded ECG images
│   ├── MI(55).jpg
│   ├── HB(209).jpg
│   └── ...
│
├── static/
│   ├── style.css
│   ├── style-login.css
│   ├── style-result.css
│   ├── style-upload.css
│   └── images/
│       ├── logo.jpg
│       ├── background.jpg
│       └── ...
│
└── templates/
    ├── login.html
    ├── index.html
    ├── upload.html
    └── result.html
```

---

## ✅ Installation Guide

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ecg-analysis-app.git
cd ecg-analysis-app
```

### 📦 2. Create a Virtual Environment (Recommended)

```bash
python3.7 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 📦 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### ▶️ 4. Run the App Locally

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 🌐 Deployment on Render

### 1️⃣ Install Gunicorn

```bash
pip install gunicorn
```

### 2️⃣ Create a `Procfile`

```text
web: gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 3️⃣ Push to GitHub

### 4️⃣ Create a Render Web Service

- Connect your GitHub repository  
- Set Python version: `3.7`  
- Set Start Command:  

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

- Click **Deploy** 🚀

---

## 🔮 Future Enhancements

- 🧠 Real-time ECG Signal Streaming and Monitoring  
- 🗃️ Database Support for Patient History  
- 🧪 More accurate models with larger and diverse datasets  
- 📈 Dashboard for multi-patient history and analytics  

---


---