# 🏥 Insurance Charges Predictor

A sleek, end-to-end Machine Learning web application built with Python and Streamlit. This project predicts the estimated annual insurance charges for an individual based on their personal profile, health details, and family structure using a pre-trained **Gradient Boosting Regressor**.

## ✨ Features

- **Wide-Screen Dashboard:** A clean, optimized, 3-column layout utilizing Streamlit's wide mode.
- **Warm UI Aesthetics:** Custom premium styling, including gradient headers, modified inputs, and a custom coral/peach global theme.
- **Accurate ML Predictions:** Powered by an optimized Gradient Boosting Regressor achieving an $R^2 \approx 0.87$.
- **Dynamic Result Insights:** Instantly flags high-risk profiles (e.g., smokers) alongside the estimated premium and user input summaries split into convenient dual-pane result blocks.

## 📁 Project Structure

- `app.py`: The main Streamlit frontend web application.
- `insurance_ml.py`: Data processing and machine learning pipeline that trains the model and generates the necessary artefact files.
- `.streamlit/config.toml`: Custom theme and color configurations for the web app layout.
- `requirements.txt`: Contains all vital Python package dependencies.
- `*.pkl`: Serialized Python artefacts containing the trained model, standard scaler, and categoric encoders.

## 🛠️ Installation & Setup

1. **Clone or Download** the repository to your local machine.
2. **Navigate** to the project directory in your terminal:
   ```bash
   cd "Data science lab"
   ```
3. **Install Dependencies** (It is recommended to use a virtual environment like Anaconda or venv):
   ```bash
   pip install -r requirements.txt
   ```
   *(Primary packages include: `streamlit`, `pandas`, `numpy`, `scikit-learn`)*

## 🔮 Usage

### 1. Train the ML Model (Optional)
If you do not have the `.pkl` files (or wish to train the model from scratch), execute the ML pipeline script. This processes the insurance dataset and outputs `best_model.pkl`, `scaler.pkl`, and `encoders.pkl`:
```bash
python insurance_ml.py
```

### 2. Run the Web Application
Start up the Streamlit frontend. Be sure your terminal is correctly navigated to the folder containing your newly created `.pkl` artefacts.
```bash
streamlit run app.py
```

### 3. Predict!
Your browser will open up (usually at `http://localhost:8501`). Simply slot in your Age, Sex, BMI, dependents, Region, and Smoking status, and hit **Predict Insurance Charges** to view your personalized insurance estimates!

## ⚠️ Notes
Smoking is consistently identified as the highest driving feature for elevated health insurance premiums within the dataset. The frontend application explicitly flags user inputs that specify smoking status with a prominent warning!
