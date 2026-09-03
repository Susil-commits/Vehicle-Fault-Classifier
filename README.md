# Vehicle Fault Classifier (VFC) 🚗⚡

An end-to-end Machine Learning and full-stack diagnostic system that answers the core root-cause diagnostic question:
> **"What type of vehicle fault is likely occurring?"**
*(instead of the basic binary question "Will it fail?")*

---

## 🏗️ Architecture

```
                      [ SAE J1979 / OBD-II Telemetry Dataset ]
                                        ↓
                         [ Preprocessing & Imputation ]
                         (Median Imputer fit on Train)
                                        ↓
                            [ Feature Engineering ]
                 (Thermal Stress, Power Demand, Voltage-Fuel Ratio,
                  Temp-Load Interaction, RPM-Load Discrepancy)
                                        ↓
                           [ Scaling & Selection ]
                   (StandardScaler + SelectKBest ANOVA F-test)
                                        ↓
                       [ Multiclass Classification ]
                         Random Forest  vs  XGBoost
                                        ↓
                            [ Model Evaluation ]
                   (98.75% Accuracy, 0.9875 Macro F1,
                    Confusion Matrix, Feature Importances)
                                        ↓
                           [ FastAPI REST Backend ]
                      (POST /classify, GET /health, /samples)
                                        ↓
                           [ Vite + React 19 UI ]
              (Dark-Mode HUD Dashboard, Live Telemetry Gauges,
               Diagnostic Trouble Codes, Multi-Class Probabilities)
```

---

## 🎯 Grounded Target Classes (Subsystems)

The classifier diagnoses five distinct automotive subsystem operating states:

| Subsystem Target | SAE Diagnostic Code | Severity | Key Telemetry Signatures |
| :--- | :--- | :--- | :--- |
| **Normal** | `P0000` | Nominal | RPM 750-3500, Temp 82-96°C, Voltage 13.6-14.4V, Fuel Pressure 40-52 PSI, Load 15-65% |
| **Cooling System** | `P0217` | **Critical** | Elevated Coolant Temp (>105°C up to 130°C), elevated thermal strain |
| **Battery/Electrical** | `P0562` | **Warning** | Alternator collapse / dead battery (<12.0V) or voltage regulator surge (>15.5V) |
| **Fuel System** | `P0087` | **Warning** | Depressed fuel rail pressure (<28 PSI starvation) or stuck regulator lockup (>65 PSI) |
| **Engine Mechanical** | `P0300` | **Critical** | Misfires, vacuum leaks, abnormal load-to-RPM disparity (high load at idle RPM) |

---

## 📊 Telemetry Input Schema (OBD-II PIDs)

```json
{
  "rpm": 3200,
  "engine_temperature": 110,
  "battery_voltage": 11.6,
  "fuel_pressure": 24,
  "engine_load": 82
}
```

### Diagnostic Output
```json
{
  "predicted_fault": "Cooling System",
  "confidence": 0.91,
  "confidence_percentage": 91.0,
  "severity": "Critical",
  "diagnostic_code": "P0217",
  "recommendation": "Engine coolant temperature critically elevated (>105°C). Inspect radiator cooling fan operation, thermostat actuation, coolant fluid level, and check for cylinder head gasket leakage.",
  "probabilities": {
    "Battery/Electrical": 0.04,
    "Cooling System": 0.91,
    "Engine Mechanical": 0.02,
    "Fuel System": 0.03,
    "Normal": 0.00
  }
}
```

---

## 🔬 ML Pipeline Highlights & Best Practices

1. **Strict Featurization Ordering**: Train/Test split (80/20 Stratified) is performed **BEFORE** fitting any transformers, imputers, scalers, or feature selectors, eliminating data leakage.
2. **Missing Value Imputation**: Handled via `SimpleImputer(strategy='median')` fit strictly on training distribution.
3. **Domain Feature Engineering**:
   - `thermal_stress`: Normalized temperature delta from 90°C target
   - `power_demand`: Workload coefficient `(RPM * Load) / 1000`
   - `voltage_fuel_ratio`: Cross-subsystem electrical-to-hydraulic interaction
   - `temp_load_interaction`: Compound thermal strain under high duty cycle
   - `rpm_load_discrepancy`: Absolute difference between actual load and expected RPM load curve (detects misfire drag)
4. **Model Comparison**:
   - **Random Forest**: 98.75% Accuracy | 0.9875 Macro F1
   - **XGBoost**: 98.75% Accuracy | 0.9875 Macro F1
   - Champion Model: **Random Forest** (selected for instant inference latency and deterministic multi-class probability calibration)
5. **Evaluation Visualizations**:
   - High-resolution confusion matrix heatmap exported to `ml/evaluation/confusion_matrix.png`
   - Normalized confusion matrix heatmap exported to `ml/evaluation/confusion_matrix_norm.png`
   - Feature importance bar chart exported to `ml/evaluation/feature_importance.png`

---

## 🚀 Quickstart Guide

### 1. Backend Setup & Training
```bash
# Clone and enter workspace
cd VFC

# Activate virtual environment
.venv\Scripts\activate

# (Optional) Re-run ML training pipeline
python ml/train.py

# Run unit & API integration tests
pytest api/test_api.py -v

# Start FastAPI server (runs on port 8001)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload
```

### 2. Frontend Dashboard Setup
```bash
cd ui
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.

---

## 🎙️ Fresh Graduate Interview Talking Points

- **Problem Framing**: *"In industrial predictive maintenance, binary classification ('will it fail?') often causes alarm fatigue. By framing this as a 5-class subsystem diagnostic problem, maintenance crews receive an immediate Diagnostic Trouble Code (DTC) and targeted inspection steps (e.g. cooling vs. alternator), reducing Mean Time to Repair (MTTR)."*
- **Preventing Data Leakage**: *"We strictly split the training and testing sets before fitting our median imputers, standard scalers, and SelectKBest feature selectors."*
- **Physics-Grounded Feature Engineering**: *"Raw sensor data lacks context; by engineering cross-domain features like Thermal Stress Index and RPM-to-Load Discrepancy, we gave the tree models direct indicators of mechanical drag and cooling inadequacy."*
- **Production Architecture**: *"FastAPI exposes a validated Pydantic schema with bounds checks, serving sub-10ms predictions to a reactive Vite/React dark-mode dashboard."*
