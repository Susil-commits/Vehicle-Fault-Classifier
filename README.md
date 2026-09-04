# Vehicle Fault Classifier (VFC) 🚗⚡

[![CI Pipeline](https://github.com/Susil-commits/Vehicle-Fault-Classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Susil-commits/Vehicle-Fault-Classifier/actions/workflows/ci.yml)
[![Live Demo UI](https://img.shields.io/badge/Live%20Demo-Vercel-000000?style=flat&logo=vercel&logoColor=white)](https://vehicle-fault-classifier.vercel.app)
[![API Status](https://img.shields.io/badge/Render-API%20Live-46E3B7?style=flat&logo=render&logoColor=white)](https://vehicle-fault-classifier-api.onrender.com/health)
[![API Docs](https://img.shields.io/badge/Swagger%20UI-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://vehicle-fault-classifier-api.onrender.com/docs)
[![Docker Ready](https://img.shields.io/badge/Docker-Containers%20Ready-2496ED?style=flat&logo=docker&logoColor=white)](#3-docker-containerized-deployment)

> **Live Deployments:**
> - **Diagnostic Dashboard (UI)**: [https://vehicle-fault-classifier.vercel.app](https://vehicle-fault-classifier.vercel.app)
> - **Inference API (Backend)**: [https://vehicle-fault-classifier-api.onrender.com](https://vehicle-fault-classifier-api.onrender.com)
> - **Interactive Swagger Docs**: [https://vehicle-fault-classifier-api.onrender.com/docs](https://vehicle-fault-classifier-api.onrender.com/docs)

An end-to-end machine learning and diagnostic web system for multi-class automotive fault classification. Instead of binary failure detection, VFC classifies specific subsystem fault conditions directly from OBD-II sensor telemetry and maps them to standard SAE J2012 Diagnostic Trouble Codes (DTCs) and official repair procedures.

---

## System Architecture

```
[ Telemetry Ingestion: SAE J1979 OBD-II Parameter IDs (PIDs) ]
                                    ↓
   [ Dataset: data/raw/vehicle_fault_dataset.csv (12,000 Records) ]
   Sensor PIDs + Ground-Truth Fault Category + SAE J2012 DTC + Repair Procedure
                                    ↓
                [ ML Training Pipeline: ml/train.py ]
   Imputation → Feature Engineering → Scaling → ANOVA Selection → GridSearchCV
   (Evaluates LightGBM, XGBoost, Random Forest, and MLP Neural Network)
                                    ↓
              [ Serialized Model Artifacts: ml/model/ ]
   best_model.pkl + scaler.pkl + imputer.pkl + fault_catalog.json
                                    ↓
             [ Diagnostic Inference API: api/predictor.py ]
   Predicts fault class; resolves SAE DTC code, severity, and procedure
                                    ↓
                  [ Supabase PostgreSQL + React HUD ]
```

---

## Automotive Diagnostic Standards

VFC aligns diagnostic telemetry and trouble codes with international automotive standards and physical operating envelopes:

| Standard | Scope | Operating Envelopes / Thresholds |
| :--- | :--- | :--- |
| **SAE J1979 / ISO 15031-5** | Diagnostic Parameter Identifiers (OBD-II PIDs) | • Engine RPM (`PID 0x0C`): 750–4200 RPM<br>• Coolant Temp (`PID 0x05`): 80–130°C<br>• Battery Voltage (`PID 0x42`): 9.8–16.5 V<br>• Fuel Rail Pressure (`PID 0x0A`): 14–78 PSI<br>• Calculated Engine Load (`PID 0x04`): 15–99 % |
| **SAE J2012 / ISO 15031-6** | Diagnostic Trouble Codes (DTCs) | • `P0000`: Nominal Operating Envelope<br>• `P0217`: Engine Coolant Overtemperature Condition<br>• `P0562`: System Voltage Low / Charging Failure<br>• `P0087`: Fuel Rail Pressure Too Low (Starvation)<br>• `P0300`: Random/Multiple Cylinder Misfire Detected |
| **Bosch Automotive Handbook (10th Ed.)** | Operating Boundaries & Failure Thresholds | • Thermostat regulation window: 82–96°C; Overheat threshold: >105°C<br>• Alternator regulated voltage: 13.6–14.4V; Undercharge threshold: <12.0V<br>• Fuel rail delivery pressure: 40–52 PSI; Starvation threshold: <28 PSI<br>• Nominal engine load: 15–65%; Misfire drag threshold: >75% load at idle (<1000 RPM) |

---

## Fault Classification & Diagnostic Mapping

| Fault Category | SAE DTC | Subsystem | Failure Signature | Standard Repair Protocol |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | `P0000` | Powertrain — Nominal | All sensor PIDs within standard operating envelope | System operating within certified tolerance. No action required. |
| **Cooling System** | `P0217` | Powertrain — Thermal Management | ECT > 105°C, high thermal-load interaction | Inspect thermostat valve opening, verify cooling fan relay activation, inspect water pump flow, and pressure test cooling circuit. |
| **Battery/Electrical** | `P0562` | Powertrain — Charging System | Voltage < 12.0V or regulator surge > 15.5V | Test alternator output under load, measure battery cold cranking amps (CCA), and inspect chassis ground connections. |
| **Fuel System** | `P0087` | Powertrain — Fuel Delivery | Fuel rail pressure < 28 PSI | Test low-pressure in-tank pump flow rate, inspect fuel filter for clogging, and verify rail pressure regulator. |
| **Engine Mechanical** | `P0300` | Powertrain — Ignition / Mechanical | Disproportionate load-to-RPM (>75% load at idle) | Inspect ignition coil pack waveform, inspect spark plug electrode gap and fouling, perform cylinder compression test, and check manifold vacuum. |

---

## ML Pipeline & Performance

The training pipeline ([ml/train.py](file:///c:/Users/nayak/OneDrive/Desktop/VFC/ml/train.py)) enforces strict featurization ordering (stratified train/test split before fitting transformers), median imputation, domain feature engineering (thermal stress, power demand, voltage-to-fuel ratio), standard scaling, ANOVA feature selection, and **3-Fold Stratified GridSearchCV** across four distinct architectures: **LightGBM**, **XGBoost**, **Random Forest**, and an **MLP Neural Network**.

### Multi-Model Benchmark Comparison (3-Fold Stratified CV & Hold-Out Test Set)

| Architecture | Model Family | Hold-Out Accuracy | Macro F1 | Macro Precision | Macro Recall | 3-Fold CV F1 | Optimal Hyperparameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** (Champion) | Fast Gradient Boosting | **99.00%** | **0.9900** | **0.9900** | **0.9900** | **0.9906** | `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200}` |
| **XGBoost** | Gradient Boosted Trees | **98.88%** | **0.9888** | **0.9888** | **0.9887** | **0.9893** | `{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}` |
| **Random Forest** | Bagging Tree Ensemble | **98.71%** | **0.9871** | **0.9872** | **0.9871** | **0.9893** | `{'max_depth': 12, 'min_samples_split': 2, 'n_estimators': 200}` |
| **MLP (Neural Network)** | Deep Feedforward NN | **98.54%** | **0.9854** | **0.9855** | **0.9854** | **0.9834** | `{'alpha': 0.0001, 'hidden_layer_sizes': (64, 32)}` |

### Architectural Insight: Why Tree Models Win on Tabular Telemetry
Automotive sensor telemetry is governed by step-function physical operating boundaries (e.g. Engine Coolant Temp > 105°C for cooling overheating, Battery Voltage < 12.0V for charging failure, Fuel Rail Pressure < 28 PSI for fuel starvation, Disproportionate Load > 75% at idle for misfire drag).

- **Tree Ensembles (LightGBM, XGBoost, Random Forest)** isolate these threshold boundaries directly through orthogonal axis-aligned splits. They are naturally invariant to feature scale, require minimal hyperparameter calibration to capture non-linear step triggers, and avoid overfitting unnormalized tabular features.
- **Neural Networks (MLP)** attempt to approximate sharp discrete thresholds using continuous smooth activation functions (ReLU, Sigmoid). This requires substantially more data, extensive feature normalization, and delicate weight regularization to avoid gradient smoothing across sharp physical boundaries.

- **Audit Script**: Verified zero data leakage and exact metric reproduction via `python ml/verify_pipeline.py`.

---

## Tech Stack

- **Machine Learning**: LightGBM, XGBoost, Scikit-Learn (Random Forest, MLPClassifier, GridSearchCV, Pipeline), NumPy, Pandas, Matplotlib, Seaborn
- **Backend API**: FastAPI, Uvicorn, Pydantic v2
- **Database Persistence**: Supabase PostgreSQL (SQLAlchemy with connection pooling and SQLite fallback)
- **Frontend Dashboard**: Vite, React 19, Vanilla CSS (Dark-Mode Automotive HUD, live telemetry gauges)
- **DevOps & Containerization**: Docker (`Dockerfile.backend`, `Dockerfile.frontend`), Docker Compose, GitHub Actions CI/CD (`.github/workflows/ci.yml`), Render, Vercel
- **Dataset**: Engineered synthetic telemetry dataset (`data/raw/vehicle_fault_dataset.csv`, 12,000 records) based on SAE J1979 PIDs and SAE J2012 DTC specifications via `data/generate_dataset.py`

---

## Quickstart Guide

### 1. Docker Containerized Deployment (Recommended)
Run the entire platform (FastAPI backend + React frontend) with a single command:
```bash
docker compose up --build
```
- Frontend UI: **`http://localhost:5173`**
- Inference API: **`http://localhost:8001`**
- Interactive Swagger Docs: **`http://localhost:8001/docs`**

### 2. Local Python & Node Setup

#### Backend & ML
```bash
# Clone repository
git clone https://github.com/Susil-commits/Vehicle-Fault-Classifier.git
cd Vehicle-Fault-Classifier

# Activate virtual environment
.venv\Scripts\activate

# Run ML verification audit
python ml/verify_pipeline.py

# Run API tests
pytest api/test_api.py -v

# Start FastAPI backend (port 8001)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload
```

#### Frontend UI
```bash
cd ui
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.

---

## API Reference

| Method | Endpoint | Description | Auth Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | API, ML model, and Supabase database connectivity status | Public |
| `POST` | `/classify` | Classifies 5-parameter OBD-II telemetry, resolves DTC & procedure, logs to DB | Public |
| `GET` | `/history` | Retrieves recent persisted diagnostic scans from Supabase PostgreSQL | Public |
| `DELETE` | `/history` | Clears stored diagnostic logs from database | **Protected** (`X-API-Key` header) |
| `GET` | `/samples` | Curated diagnostic scenario presets for 1-click live testing | Public |
| `GET` | `/model-info` | Returns model architecture, CV metrics, multi-model benchmark, and feature list | Public |
| `GET` | `/confusion-matrix` | Serves the generated multi-class confusion matrix heatmap image | Public |

---

## Known Limitations & Production Roadmap

This system is engineered as an automotive diagnostic demonstration and reference architecture:

1. **Authentication & Authorization (Resolved for Mutating Endpoints)**:
   - Destructive operations (`DELETE /history`) are protected with header-based API key authentication (`X-API-Key`), preventing unauthorized data purging while allowing seamless developer and administrative management via environment variables (`VFC_API_KEY`).
   - *Enterprise Roadmap*: Integrate OAuth2 / JWT bearer tokens with role-based access control (RBAC) to differentiate fleet technician roles.

2. **CORS Policy & Origin Isolation**:
   - The API is configured with `allow_origins=["*"]` and `allow_credentials=False`, strictly adhering to W3C CORS specifications. In enterprise production deployments, `allow_origins` would be locked down to explicitly whitelisted telemetry portal domains.

3. **Telemetry Realism & CAN-Bus Validation**:
   - The primary dataset is synthetically generated via `data/generate_dataset.py` from deterministic SAE J1979/J2012 physical operating boundaries and Bosch Automotive Handbook thresholds.
   - *Roadmap*: The reserved `data/raw/EngineFaultDB_Final.csv` benchmark and live OBD-II vehicle logging benches serve as the next phase for out-of-distribution robustness evaluation against noisy CAN-bus arbitration and sensor degradation.
