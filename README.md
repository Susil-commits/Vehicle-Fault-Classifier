# Vehicle Fault Classifier (VFC) 🚗⚡

An end-to-end Machine Learning and full-stack diagnostic system addressing the core root-cause diagnostic question:
> **"What type of vehicle fault is likely occurring?"**
*(instead of the basic binary question "Will it fail?")*

---

## 🎯 Domain Grounding: No "Fake AI" or Fabricated Semantics

A common weakness in ML projects is disconnected labels and hardcoded mock outputs. In VFC, the entire pipeline is **100% data-driven and grounded in international automotive engineering standards**:

```
[ Domain Provenance: SAE J1979 OBD-II PIDs & Bosch Automotive Handbook (10th Ed.) ]
                                        ↓
       [ Dataset (data/raw/vehicle_fault_dataset.csv - 12,000 Records) ]
       Every row contains: Raw Sensor PIDs + Ground-Truth Fault Category +
       SAE J2012 Diagnostic Trouble Code + Official Diagnostic Procedure
                                        ↓
                     [ ML Training Pipeline (ml/train.py) ]
       Extracts Ground-Truth Fault Catalog directly from dataset columns;
       Trains Random Forest & XGBoost on hold-out stratified splits
                                        ↓
                  [ Trained Model Artifacts (ml/model/) ]
       best_model.pkl + scaler.pkl + imputer.pkl + fault_catalog.json
                                        ↓
                  [ Diagnostic Inference (api/predictor.py) ]
       Predicts fault category; dynamically retrieves exact SAE J2012 DTC,
       subsystem title, and repair protocol directly from fault_catalog.json
                                        ↓
                      [ Supabase PostgreSQL + React HUD ]
```

---

## 📑 Automotive Engineering Standards Backing

| Standard | Application in This System | Physical Operational Envelope / Triggering Threshold |
| :--- | :--- | :--- |
| **SAE J1979 / ISO 15031-5** | Diagnostic Parameter Identifiers (OBD-II PIDs) | • Engine RPM (`PID 0x0C`): 750–4200 RPM<br>• Coolant Temp (`PID 0x05`): 80–130°C<br>• Battery Voltage (`PID 0x42`): 9.8–16.5 V<br>• Fuel Rail Pressure (`PID 0x0A`): 14–78 PSI<br>• Calculated Engine Load (`PID 0x04`): 15–99 % |
| **SAE J2012 / ISO 15031-6** | Ground-Truth Diagnostic Trouble Codes (DTCs) | • `P0000`: Nominal Operating Envelope<br>• `P0217`: Engine Coolant Overtemperature Condition<br>• `P0562`: System Voltage Low / Charging Failure<br>• `P0087`: Fuel Rail Pressure Too Low (Starvation)<br>• `P0300`: Random/Multiple Cylinder Misfire Detected |
| **Bosch Automotive Handbook (10th Edition)** | Physics & Failure Thresholds | • Thermostat regulating window: 82–96°C; Overheat trip: >105°C<br>• Alternator regulated voltage: 13.6–14.4V; Undercharge trip: <12.0V<br>• Fuel delivery rail pressure: 40–52 PSI; Starvation trip: <28 PSI<br>• Engine load duty: 15–65%; Misfire drag trip: >75% load at idle (<1000 RPM) |

---

## 🏛️ Traceability Matrix: Dataset → Target → Feature → Prediction → Recommendation

| Ground-Truth Class | Ground-Truth DTC | Ground-Truth Subsystem | Failure Signature | Standard Repair Protocol (from Dataset) |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | `P0000` | Powertrain — Nominal | All sensor PIDs within standard operating envelope | Certified within operational tolerance. No technician action required. |
| **Cooling System** | `P0217` | Powertrain — Thermal Management | ECT > 105°C, high thermal load interaction | Execute SAE J2012 cooling diagnostic tree: inspect thermostat opening valve, verify radiator cooling fan relay, water pump flow, and pressure test for head gasket breach. |
| **Battery/Electrical** | `P0562` | Powertrain — Charging System | Voltage < 12.0V (or regulator surge > 15.5V) | Execute SAE J2012 charging system test: test alternator output under load, measure battery cold cranking amps (CCA), and inspect ground loops. |
| **Fuel System** | `P0087` | Powertrain — Fuel Delivery | Fuel rail pressure < 28 PSI | Execute SAE J2012 fuel delivery diagnostic: test low-pressure in-tank fuel pump delivery rate, inspect fuel filter for particulate clogging, and inspect rail pressure regulator. |
| **Engine Mechanical** | `P0300` | Powertrain — Ignition / Mechanical | Extreme load-to-RPM disparity (>75% load at idle) | Execute SAE J2012 misfire diagnostic procedure: inspect ignition coil pack waveform, inspect spark plug electrode gap and carbon fouling, perform cylinder compression test, and check manifold vacuum. |

---

## 🔬 ML Pipeline Performance & Audit

Trained with strict featurization ordering (stratified train/test split **before** fitting transformers), median imputation, domain feature engineering, standard scaling, and ANOVA feature selection.

### Verified Hold-Out Test Metrics (2,400 Samples)
- **Overall Accuracy**: **98.75%** (2,370 correct out of 2,400)
- **Macro Precision**: **0.9876**
- **Macro Recall**: **0.9875**
- **Macro F1-Score**: **0.9875**
- **5-Fold Stratified CV Macro F1**: **0.9880 (± 0.0037)**
- **Zero Data Leakage**: Audited and confirmed by `python ml/verify_pipeline.py`.

---

## 🛠️ Tech Stack & Architecture

- **Machine Learning**: Scikit-Learn (Random Forest & XGBoost), NumPy, Pandas, Matplotlib, Seaborn
- **Backend API**: FastAPI, Uvicorn, Pydantic v2
- **Database Persistence**: Supabase PostgreSQL (SQLAlchemy with connection pooling)
- **Frontend Dashboard**: Vite, React 19, Vanilla CSS (Dark-Mode Automotive HUD, live telemetry gauges)
- **Benchmarking Dataset**: Published IEEE C14NE laboratory dataset (`data/raw/EngineFaultDB_Final.csv`)

---

## 🚀 Quickstart Guide

### 1. Backend & ML
```bash
# Clone repository
git clone https://github.com/Susil-commits/Vehicle-Fault-Classifier.git
cd Vehicle-Fault-Classifier

# Activate virtual environment
.venv\Scripts\activate

# Run verification audit
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

---

## 🎙️ Fresh Graduate Interview Talking Points

- **Answering "Where did these labels come from?"**:
  *"Rather than inventing arbitrary labels or hardcoding mock recommendations, our dataset is grounded in SAE J1979 OBD-II standard PIDs (0x0C, 0x05, 0x42, 0x0A, 0x04) and SAE J2012 Diagnostic Trouble Code specifications. The operational thresholds (e.g. 105°C coolant trip for P0217, 12.0V charging collapse for P0562) follow published figures from the Bosch Automotive Handbook (10th Ed.). The ML training pipeline extracts the ground-truth fault catalog directly from the dataset columns and exports it alongside the serialized model, ensuring full data-driven provenance with zero fabricated semantics."*
- **Preventing Data Leakage**:
  *"We strictly split the dataset into train and test sets before fitting the median imputer, standard scaler, and ANOVA feature selector. Our independent audit script `ml/verify_pipeline.py` asserts zero index overlap and verifies that imputer and scaler statistics match the training fold exclusively."*
- **Full-Stack Persistence**:
  *"Predictions don't vanish upon UI refresh. Every diagnostic inference is logged via SQLAlchemy to a hosted Supabase PostgreSQL instance, allowing historical telemetry trends to be reloaded and reviewed."*
