"""
Inference Engine and Diagnostic Mapping for Vehicle Fault Classifier
Loads serialized preprocessing and ML artifacts to perform low-latency fault classification.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


class VehicleFaultPredictor:
    _instance: Optional["VehicleFaultPredictor"] = None

    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            # Default to ml/model relative to project root
            base_path = Path(__file__).resolve().parent.parent
            model_dir = base_path / "ml" / "model"

        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.label_encoder = None
        self.metadata = {}
        self.load_artifacts()

    @classmethod
    def get_instance(cls) -> "VehicleFaultPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_artifacts(self):
        """Loads all serialized scikit-learn & metadata artifacts."""
        model_file = self.model_dir / "best_model.pkl"
        scaler_file = self.model_dir / "scaler.pkl"
        imputer_file = self.model_dir / "imputer.pkl"
        selector_file = self.model_dir / "feature_selector.pkl"
        encoder_file = self.model_dir / "label_encoder.pkl"
        meta_file = self.model_dir / "model_metadata.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Model artifact missing at {model_file}. Run ml/train.py first.")

        with open(model_file, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_file, "rb") as f:
            self.scaler = pickle.load(f)
        with open(imputer_file, "rb") as f:
            self.imputer = pickle.load(f)
        with open(selector_file, "rb") as f:
            self.feature_selector = pickle.load(f)
        with open(encoder_file, "rb") as f:
            self.label_encoder = pickle.load(f)

        if meta_file.exists():
            with open(meta_file, "r") as f:
                self.metadata = json.load(f)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mirror the exact feature engineering performed during training."""
        df_feat = df.copy()
        df_feat["thermal_stress"] = (df_feat["engine_temperature"] - 90.0) / 10.0
        df_feat["power_demand"] = (df_feat["rpm"] * df_feat["engine_load"]) / 1000.0
        df_feat["voltage_fuel_ratio"] = df_feat["battery_voltage"] / (df_feat["fuel_pressure"] + 1e-5)
        df_feat["temp_load_interaction"] = (df_feat["engine_temperature"] * df_feat["engine_load"]) / 100.0
        expected_load = (df_feat["rpm"] / 4200.0) * 50.0
        df_feat["rpm_load_discrepancy"] = np.abs(df_feat["engine_load"] - expected_load)
        return df_feat

    def predict(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """
        Runs the end-to-end inference pipeline on raw telemetry input.
        Returns predicted fault, confidence score, all probabilities, DTC code, and recommendation.
        """
        raw_cols = ["rpm", "engine_temperature", "battery_voltage", "fuel_pressure", "engine_load"]
        df_input = pd.DataFrame([telemetry], columns=raw_cols)

        # 1. Median Imputation (safety fallback if any NaN)
        df_imputed = pd.DataFrame(self.imputer.transform(df_input), columns=raw_cols)

        # 2. Feature Engineering
        df_engineered = self._engineer_features(df_imputed)

        # 3. Standardization
        scaled_features = self.scaler.transform(df_engineered)

        # 4. Feature Selection
        selected_features = self.feature_selector.transform(scaled_features)

        # 5. Model Inference (Probabilities)
        probabilities = self.model.predict_proba(selected_features)[0]
        classes = self.label_encoder.classes_

        prob_dict = {str(cls_name): round(float(prob), 4) for cls_name, prob in zip(classes, probabilities)}

        # Champion prediction
        best_idx = int(np.argmax(probabilities))
        predicted_fault = str(classes[best_idx])
        confidence = float(probabilities[best_idx])

        # Domain Diagnostics & Recommendations
        dtc_map = {
            "Normal": {
                "code": "P0000",
                "severity": "Normal",
                "recommendation": "Vehicle telemetry is within nominal operating range. No active DTCs or system faults detected.",
            },
            "Cooling System": {
                "code": "P0217",
                "severity": "Critical",
                "recommendation": "Engine coolant temperature critically elevated (>105°C). Inspect radiator cooling fan operation, thermostat actuation, coolant fluid level, and check for cylinder head gasket leakage.",
            },
            "Battery/Electrical": {
                "code": "P0562",
                "severity": "Warning",
                "recommendation": "Abnormal charging system voltage detected (<12.0V or >15.5V). Test alternator diode ripple, evaluate battery internal resistance, check drive belt tension, and inspect terminal corrosion.",
            },
            "Fuel System": {
                "code": "P0087",
                "severity": "Warning",
                "recommendation": "Fuel rail pressure out of specification (<28 psi or rich lockup). Inspect fuel delivery pump flow rate, fuel filter clogging, fuel rail pressure sensor, and injector spray pattern.",
            },
            "Engine Mechanical": {
                "code": "P0300",
                "severity": "Critical",
                "recommendation": "Abnormal load-to-RPM disparity detected, indicating cylinder misfire or mechanical drag. Inspect ignition coil packs, spark plug fouling, cylinder compression balance, and intake vacuum leaks.",
            },
        }

        diag_info = dtc_map.get(
            predicted_fault,
            {
                "code": "P0999",
                "severity": "Caution",
                "recommendation": "General anomaly detected. Run full OBD-II scan diagnostic.",
            },
        )

        return {
            "predicted_fault": predicted_fault,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 1),
            "severity": diag_info["severity"],
            "diagnostic_code": diag_info["code"],
            "recommendation": diag_info["recommendation"],
            "probabilities": prob_dict,
            "telemetry_received": telemetry,
        }
