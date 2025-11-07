# 😃 Emotion Recognition from Facial Expressions and Text  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)

## 📁 Emotion Recognition Project Structure

emotion_detection_project/
│
├── 📄 README.md                         # Project documentation (you just created this)
│
├── 🧠 models/                           # Trained models for both modalities
│   ├── image_emotion.h5                 # CNN model for facial emotion recognition
│   └── text_emotion/
│       └── pipeline.joblib              # Trained NLP pipeline for text emotion classification
│
├── 📂 data/                             # Datasets used for training
│   ├── images/
│   │   └── fer2013/
│   │       ├── train/
│   │       │   ├── angry/
│   │       │   ├── disgust/
│   │       │   ├── fear/
│   │       │   ├── happy/
│   │       │   ├── neutral/
│   │       │   ├── sad/
│   │       │   └── surprise/
│   │       ├── test/
│   │       └── validation/
│   │
│   └── text/
│       ├── train.txt                    # Training data (text + emotion)
│       ├── val.txt                      # Validation data
│       └── test.txt                     # Testing data
│
├── 🧩 src/                              # Source Python scripts
│   ├── train_image.py                   # Trains the CNN model on FER2013 dataset
│   ├── train_text.py                    # Trains the text emotion classification pipeline
│   ├── webcam_infer.py                  # Optional script for direct webcam testing
│   └── multimodal_server.py             # Core backend server for browser communication (no Flask/FastAPI)
│
├── 🌐 web_demo/                         # Frontend web files (user interface)
│   ├── index.html                       # Beautiful sky-blue themed UI (text + webcam detection)
│   └── assets/                          # (Optional) for CSS, JS, or icons (if needed later)
│
├── 🧰 venv/                             # Virtual environment (Python dependencies)
│
├── 📜 requirements.txt                  # (Optional) List of required dependencies
│
└── ⚙️ .gitignore                        # (Optional) Ignore venv, __pycache__, etc.

---

## 🧠 1. Project Title and Description

### **Emotion Recognition System (Facial & Text-based)**  

This project detects human emotions from **facial expressions (via webcam)** and **text input** using a combination of **deep learning (CNN)** and **machine learning (NLP pipeline)** models.  
It provides real-time emotion analysis with a beautiful, responsive web interface — without using Flask or FastAPI.  

#### 🎯 **Purpose**
To create an AI system that understands human emotional states for use in:
- Human-Computer Interaction (HCI)
- Sentiment-based feedback systems
- Smart education and healthcare systems

#### ✨ **Key Highlights**
- Real-time facial emotion detection 🎥  
- Text emotion detection 💬  
- Confidence rings and emoji indicators 😃😢😡  
- Emotion history tracking with clear option  
- Sky-blue, dark/light mode UI 🌗  

---

## 📑 2. Table of Contents
- [1. Project Title and Description](#-1-project-title-and-description)
- [2. Table of Contents](#-2-table-of-contents)
- [3. Installation Instructions](#-3-installation-instructions)
- [4. Usage Instructions](#-4-usage-instructions)
- [5. Features](#-5-features)
- [6. Technologies Used](#-6-technologies-used)
- [7. Contributing Guidelines](#-7-contributing-guidelines)
- [8. License](#-8-license)
- [9. Credits](#-9-credits)
- [10. Contact Information](#-10-contact-information)
- [Known Issues](#known-issues)
- [Future Plans](#future-plans)

---

## ⚙️ 3. Installation Instructions

### 🧩 **Prerequisites**
- Python 3.8 or higher  
- Camera access for facial detection  
- FER-2013 dataset for images  
- Kaggle text emotion dataset  

---

### 🧠 **Setup Steps**

```bash
# 1️⃣ Create project environment
python -m venv venv

# 2️⃣ Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 3️⃣ Upgrade pip and install dependencies
pip install --upgrade pip
pip install numpy pandas opencv-python matplotlib scikit-learn joblib tensorflow==2.12
pip install pillow Werkzeug aiohttp
```

---

## ▶️ 4. Usage Instructions

### 📁 **Dataset Structure**
```
data/
├── images/
│   └── fer2013/
│       ├── train/
│       ├── test/
│       └── validation/
└── text/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

---

### 🧠 **Training Models**

```bash
# Train CNN on FER-2013 images
python src/train_image.py

# Train NLP model on text emotions
python src/train_text.py
```

✅ Creates:
```
models/
├── image_emotion.h5
└── text_emotion/
    └── pipeline.joblib
```

---

### 🚀 **Start Backend Server**

```bash
python src/multimodal_server.py
```

You should see:
```
Loading models...
✅ Server running at http://localhost:8000
```

---

### 🌐 **Open Web Interface**
Visit:
👉 [http://localhost:8000](http://localhost:8000)

---

### 💬 **Text Emotion**
- Type text (e.g., *"I feel amazing today!"*)  
- Click **Analyze Text**  
- View:
  - Predicted emotion with emoji 😍  
  - Confidence ring  
  - Probability bars  
  - Entry in **Emotion History**

---

### 📷 **Facial Emotion**
- Allow webcam access  
- Click **▶️ Start Live Detection**  
- Watch real-time facial emotion recognition  
- Logs each detection to **History**

---

### 🧹 **Manage History**
- All detections (text & image) are logged with timestamp  
- Click **🧹 Clear History** to remove them  

---

## 🌟 5. Features

| Feature | Description |
|----------|--------------|
| 💬 Text Emotion Detection | NLP-based emotion prediction |
| 📷 Facial Emotion Detection | CNN-based webcam analysis |
| 🧠 Dual AI Pipeline | Combines ML + DL techniques |
| 🧾 Prediction History | Tracks all past detections |
| 🧹 Clear Button | Clears log instantly |
| 🌗 Dark/Light Mode | Theme toggle |
| 🎨 Modern UI | Sky-blue gradient with glassmorphism |
| 💻 Laptop Friendly | Responsive layout for 13–15" screens |

---

## 🧰 6. Technologies Used

| Category | Tools |
|-----------|-------|
| **Language** | Python 3 |
| **ML/DL** | TensorFlow, Keras, scikit-learn |
| **CV/NLP** | OpenCV, TF/Keras, Joblib |
| **Data Handling** | Pandas, NumPy |
| **Frontend** | HTML, CSS, JavaScript |
| **Server** | Python `http.server` |
| **Visualization** | Matplotlib, custom JS progress bars |

---

## 🤝 7. Contributing Guidelines

Contributions are welcome! 🎉  

1. Fork the repo  
2. Create a new branch (`feature-name`)  
3. Commit your changes  
4. Push and open a Pull Request  

You can also:
- 🐞 Report issues  
- 💡 Suggest new features  
- 🧠 Improve UI/UX  

---

## 📜 8. License

This project is licensed under the **MIT License**.  

```
MIT License © 2025 Nagateja Goud
```

---

## 🙌 9. Credits

**Developed by:** [Nagateja Goud](#)  

**Datasets:**
- FER-2013 (Kaggle)
- Text Emotion Dataset (Kaggle NLP Dataset)

**Libraries Used:**
TensorFlow • scikit-learn • OpenCV • NumPy • Pandas • Joblib  

---

## 📩 10. Contact Information

💻 **GitHub:** [github.com/nagateja8185](https://github.com/nagateja8185)  
🌐 **LinkedIn:** [linkedin.com/in/thimmapur-nagateja-goud8185](www.linkedin.com/in/thimmapur-nagateja-goud8185)

---

## 🐛 Known Issues

- Low-light webcam conditions may reduce facial accuracy.  
- Webcam capture rate depends on browser permissions.  
- Dataset imbalance can affect emotion confidence.  

---

## 🚀 Future Plans

- 🎤 Add **Voice Emotion Recognition (audio)**  
- 👥 Detect multiple faces simultaneously  
- ☁️ Deploy on **Streamlit or WebApp**  
- 📈 Add **live emotion trend charts**  

---

### 🧭 Quick Command Reference

```bash
# Create venv
python -m venv venv

# Activate
venv\Scripts\activate   # (Windows)
source venv/bin/activate  # (Linux/macOS)

# Install dependencies
pip install --upgrade pip
pip install numpy pandas opencv-python matplotlib scikit-learn joblib tensorflow==2.12 pillow Werkzeug aiohttp

# Train models
python src/train_image.py
python src/train_text.py

# Start backend server
python src/multimodal_server.py

# Open in browser
http://localhost:8000
```

---

⭐ **If you found this project useful, please give it a star on GitHub!** 🌟
