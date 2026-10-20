# 🧠 Parkinson's Disease Detection Platform

An intelligent, end-to-end clinical system for early detection of Parkinson's Disease through speech signal analysis. Built with Python, Streamlit, Machine Learning (SVM, Random Forest, CNN), and **Cerebras LLaMA 3.1 8B AI** for personalized clinical recommendations.

---

## 🌟 Key Features

- **🔐 Secure Authentication:** User registration and login system with encrypted credentials.
- **📊 AI Patient Dashboard:** Personalized prediction history, risk distribution charts, and dynamic AI-generated trend summaries based on past results.
- **🎤 Live Voice Recording:** Record audio directly from the browser for instant acoustic analysis.
- **🤖 Cerebras AI Integration:** Uses the blazing fast Cerebras Cloud SDK and LLaMA 3.1 8B to generate dynamic Precautions, Dos & Don'ts, Diet Plans, and follow-up doctor questions based on risk level.
- **💬 AI Neurologist Chat:** An interactive assistant integrated directly into the results page to answer patient questions on the fly.
- **📈 Advanced ML Engine:** GridSearchCV-optimized SVM, Random Forest, and Deep 1D-CNN trained on 5,875+ clinical voice samples across 252 subjects.

---

## 📁 Project Structure

```
PARKINSON_Project/
├── app.py                     # Streamlit web application (main entry point)
├── auth.py                    # SQLite authentication & history storage
├── ai_engine.py               # Cerebras LLaMA 3.1 API integration
├── voice_handler.py           # Audio widget and simulated DSP feature extraction
├── utils.py                   # Helper functions (prediction, dataset loading)
├── style.css                  # Premium dark theme styling
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── PROJECT_REPORT.md          # Full B.Tech project report document
├── user_credentials.txt       # Default test login instructions
├── data/                      # Datasets & SQLite databases
└── models/
    ├── train_models.py        # Model training pipeline
    └── saved/                 # Trained model artifacts
```

---

## 🛠️ Prerequisites

- **Python 3.9+** installed on your system
- **pip** package manager

---

## 🚀 How to Run (End-to-End)

### Step 1: Clone / Download the Project

Download the `PARKINSON_Project` folder to your local machine.

### Step 2: Open Terminal

Navigate to the project folder:

```bash
cd path/to/PARKINSON_Project
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch the Web Application

The models come pre-trained. Simply start the app:

```bash
streamlit run app.py
```

The app will open in your browser at **http://localhost:8502** (or 8501).

---

## 📄 Application Pages

### 🔐 Login / Register
Create a new account or log in with the test credentials provided in `user_credentials.txt` to track your diagnostic history securely.

### 📊 Dashboard
View all your past voice analyses. Features a detailed history table and an interactive risk distribution pie chart (High/Medium/Low Risk).

### 🏠 Home
Project overview and description, key metrics (5,875+ training samples, 92.3% accuracy), and the technology pipeline diagram.

### 🎤 Voice Recorder
Record your voice (prolonged 'aah' vowel) via the microphone or upload a WAV/MP3 file. The system extracts deterministic simulated acoustic features based on the audio profile to demonstrate the Parkinson's risk diagnosis pipeline.

### 🔬 Predict
Manually enter the 22 biomedical speech features or upload a batch CSV for processing. Instantly outputs the confidence score and **Cerebras AI Clinical Recommendations**.

### 📋 AI Results Page
For every prediction, the **Cerebras LLaMA 3.1 8B** engine calculates the risk level and dynamically generates:
- 🛡️ Personalized medical precautions
- ✅❌ Dos & Don'ts
- 🥗 Tailored dietary recommendations
- ❓ Custom follow-up questions to ask a doctor
- 💬 Direct conversational interface to ask the AI Neurologist further questions.

### 📈 Model Analytics
Compare the accuracy, precision, recall, and F1 scores of the 3 underlying ML models via interactive Plotly charts, confusion matrices, and ROC curves.

### 📖 About
Full project abstract, ML model descriptions, dataset specifics, and team member information.

---

## 📊 Dataset Specifications

The ML models are trained on a comprehensive **multi-source clinical dataset**:
- **Source:** Combinations of multiple UCI Machine Learning Repositories (Parkinsons Dataset, Telemonitoring, Multiple Recordings).
- **Samples:** 5,875+ voice recordings from 252 subjects.
- **Features:** 22 biomedical speech features.

## Technologies and Frameworks:
  ***Python, Pandas, NumPy, scikit-learn, random forest classifier, patient voice recordings with extracted vocal features, 
  CSV files (database, not confirmed), Git, and GitHub.

---

## ⚠️ Disclaimer

This system is designed for **research and educational purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for any medical concerns.
