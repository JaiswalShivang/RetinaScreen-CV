# 👁️ OculAI: Retinal Disease Screener

OculAI is a premium, AI-powered diagnostic screening web dashboard for Optical Coherence Tomography (OCT) retinal scans. The system performs real-time classification across four major clinical categories and generates structural focus heatmaps utilizing Grad-CAM to visualize and localize abnormalities.

---

## ✨ Features

- 🏥 **Diagnostic Screening**: Automatically classifies retinal scans into one of four clinical categories:
  - **CNV (Choroidal Neovascularization)**: Subretinal fluid and membrane growth.
  - **DME (Diabetic Macular Edema)**: Macular fluid accumulation.
  - **DRUSEN**: Early age-related macular degeneration (AMD) indicators.
  - **NORMAL**: Healthy retinal structures.
- 🔍 **Grad-CAM Localization Maps**: Generates class-activation mapping to highlight regions of interest (e.g., fluid pockets, layer disruptions) that influenced the network's classification.
- 🧪 **Instant Testing Sandbox**: Generate simulated OCT scans programmatically to test CNV, DME, DRUSEN, or NORMAL pathologies without needing external files.
- 📊 **Detailed Diagnostics & Guidance**: Renders visual confidence metrics and structured action plans/clinical recommendation boards.

---

## 🛠️ Architecture & Technologies

- **Frontend Interface**: [Streamlit](https://streamlit.io/) with a custom medical dark theme.
- **Deep Learning Core**: Keras & TensorFlow (EfficientNetB3 backbone).
- **Image Processing**: OpenCV, NumPy, and PIL.

---

## 🚀 Getting Started

### 📋 Prerequisites

Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Python 3.10+ installed.

### ⚙️ Setup Environment

1. Activate your conda environment or create a new one:
   ```bash
   conda activate oculai_env
   ```

2. Alternatively, install the dependencies directly via `pip`:
   ```bash
   pip install streamlit tensorflow opencv-python pillow numpy
   ```

3. Ensure the model weights file `best_oculai_model.keras` is placed in the root directory.

### 🏃 Running the Application

Launch the Streamlit dashboard by running:
```bash
streamlit run app.py
```

The application will start, and you can view it in your browser at:
**[http://localhost:8501](http://localhost:8501)**

---

## 📂 Project Structure

```
ml/
├── app.py                     # Streamlit application UI & Grad-CAM pipeline
├── best_oculai_model.keras    # Pre-trained CNN model weights (EfficientNetB3)
├── README.md                  # Documentation
└── .gitignore                 # Excluded directories and file patterns
```
