# Vehicle Fault Classifier (VFC) 🚗⚡

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

The training pipeline ([ml/train.py](file:///c:/Users/nayak/OneDrive/Desktop/VFC/ml/train.py)) enforces strict featurization ordering (stratified train/test split before fitting transformers), median imputation, domain feature engineering (thermal stress, power demand, voltage-to-fuel ratio), standard scaling, ANOVA feature selection, and **3-Fold Stratified GridSearchCV** across Random Forest and XGBoost.

### Champion Model: Hyperparameter-Tuned XGBoost
- **Optimal Hyperparameters**: `{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}`
- **Hold-Out Test Accuracy**: **98.88%** (2,373 / 2,400 correct)
- **Macro Precision**: **0.9888**
- **Macro Recall**: **0.9887**
- **Macro F1-Score**: **0.9888**
- **Weighted F1-Score**: **0.9888**
- **5-Fold Stratified CV Macro F1**: **0.9881 (± 0.0020)**
- **Audit Script**: Verified zero data leakage and exact metric reproduction via `python ml/verify_pipeline.py`.

### Dataset & Evaluation Context
The dataset (`data/raw/vehicle_fault_dataset.csv`, 12,000 records) is synthetically generated via `data/generate_dataset.py` using physical operating envelopes and threshold triggers from SAE J1979 and the Bosch Automotive Handbook. Because ground-truth labels are derived from deterministic physical boundary conditions, these metrics reflect pipeline correctness in isolating known failure signatures. Further development focuses on testing against raw, noisy CAN-bus logs and sensor degradation.

---

## Tech Stack

- **Machine Learning**: Scikit-Learn (Random Forest, GridSearchCV, Pipeline), XGBoost, NumPy, Pandas, Matplotlib, Seaborn
- **Backend API**: FastAPI, Uvicorn, Pydantic v2
- **Database Persistence**: Supabase PostgreSQL (SQLAlchemy with connection pooling)
- **Frontend Dashboard**: Vite, React 19, Vanilla CSS (Dark-Mode Automotive HUD, live telemetry gauges)
- **Dataset**: Engineered synthetic telemetry dataset (`data/raw/vehicle_fault_dataset.csv`, 12,000 records) based on SAE J1979 PIDs and SAE J2012 DTC specifications via `data/generate_dataset.py`

---

## Quickstart Guide

### 1. Backend & ML
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

### 2. Frontend UI
```bash
cd ui
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.
