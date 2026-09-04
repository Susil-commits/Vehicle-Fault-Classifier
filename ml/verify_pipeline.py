"""
ML Claims Verification & Leakage Audit Script
Validates:
1. Strict Featurization Ordering & Zero Data Leakage
2. Exact Test Performance Reproduction against Trained Model Metadata (98.88% Accuracy & 0.9888 Macro F1)
3. 5-Fold Cross Validation Stability on Training Set
4. Detailed Confusion Matrix & Misclassification Error Breakdown
"""

import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat["thermal_stress"] = (df_feat["engine_temperature"] - 90.0) / 10.0
    df_feat["power_demand"] = (df_feat["rpm"] * df_feat["engine_load"]) / 1000.0
    df_feat["voltage_fuel_ratio"] = df_feat["battery_voltage"] / (df_feat["fuel_pressure"] + 1e-5)
    df_feat["temp_load_interaction"] = (df_feat["engine_temperature"] * df_feat["engine_load"]) / 100.0
    expected_load = (df_feat["rpm"] / 4200.0) * 50.0
    df_feat["rpm_load_discrepancy"] = np.abs(df_feat["engine_load"] - expected_load)
    return df_feat


def run_verification():
    print("=" * 80)
    print("  [AUDIT] VEHICLE FAULT CLASSIFIER - ML CLAIMS & LEAKAGE VERIFICATION")
    print("=" * 80)

    # 1. Load Raw Dataset
    data_path = Path("data/raw/vehicle_fault_dataset.csv")
    assert data_path.exists(), f"Raw data not found at {data_path}"
    df = pd.read_csv(data_path)
    print(f"\n[Step 1] Dataset Loaded:")
    print(f"  Total records: {len(df)}")
    print(f"  Target column: 'fault_type' with classes {list(df['fault_type'].unique())}")

    raw_features = ["rpm", "engine_temperature", "battery_voltage", "fuel_pressure", "engine_load"]
    X_raw = df[raw_features].copy()
    y_raw = df["fault_type"].copy()

    # Load artifacts
    model_dir = Path("ml/model")
    with open(model_dir / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_dir / "imputer.pkl", "rb") as f:
        imputer = pickle.load(f)
    with open(model_dir / "feature_selector.pkl", "rb") as f:
        selector = pickle.load(f)
    with open(model_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    y_encoded = label_encoder.transform(y_raw)

    # 2. Audit Train / Test Split
    print(f"\n[Step 2] Auditing Train/Test Split & Leakage Controls:")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )

    # Index overlap verification
    train_indices = set(X_train_raw.index)
    test_indices = set(X_test_raw.index)
    overlap = train_indices.intersection(test_indices)
    assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} overlapping indices!"
    print("  [PASSED] Index Disjointness: Zero index overlap between Train (9,600) and Test (2,400).")

    # Audit Imputer stats
    expected_medians = X_train_raw.median().values
    imputer_stats = imputer.statistics_
    np.testing.assert_almost_equal(imputer_stats, expected_medians, decimal=5)
    print("  [PASSED] Imputer Leakage Check: Imputer statistics match training set medians exactly.")

    # Audit Scaler parameters
    X_train_imp = pd.DataFrame(imputer.transform(X_train_raw), columns=raw_features, index=X_train_raw.index)
    X_train_eng = engineer_features(X_train_imp)
    expected_means = X_train_eng.mean().values
    expected_stds = X_train_eng.std(ddof=0).values
    np.testing.assert_almost_equal(scaler.mean_, expected_means, decimal=4)
    np.testing.assert_almost_equal(scaler.scale_, expected_stds, decimal=4)
    print("  [PASSED] Scaler Leakage Check: StandardScaler mean & scale match training set exactly.")

    # 3. Independent Preprocessing & Feature Engineering on Test Set
    print(f"\n[Step 3] Preprocessing and Transforming Test Set:")
    X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=raw_features, index=X_test_raw.index)
    X_test_eng = engineer_features(X_test_imp)
    X_test_scaled = scaler.transform(X_test_eng)
    X_test_selected = selector.transform(X_test_scaled)
    print(f"  Test transformed shape: {X_test_selected.shape}")

    # 4. Model Prediction & Metric Verification
    print(f"\n[Step 4] Running Independent Model Inference on Test Set:")
    y_pred = model.predict(X_test_selected)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    prec_weighted = precision_score(y_test, y_pred, average="weighted")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    rec_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # Load claimed metrics from model_metadata.json
    metadata_file = model_dir / "model_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        claimed_acc = metadata.get("test_accuracy", 0.9888)
        claimed_f1 = metadata.get("test_f1_score", 0.9888)
        claimed_model_name = metadata.get("model_name", "Trained Model")
    else:
        claimed_acc = 0.9888
        claimed_f1 = 0.9888
        claimed_model_name = "Trained Model"

    print("\n" + "-" * 60)
    print(f"  METRIC VERIFICATION TABLE ({claimed_model_name})")
    print("-" * 60)
    print(f"  Test Accuracy:        {acc * 100:.2f}%  (Claimed: {claimed_acc * 100:.2f}%)")
    print(f"  Macro Precision:      {prec_macro:.4f}")
    print(f"  Macro Recall:         {rec_macro:.4f}")
    print(f"  Macro F1-Score:       {f1_macro:.4f}  (Claimed: {claimed_f1:.4f})")
    print(f"  Weighted F1-Score:    {f1_weighted:.4f}")
    print("-" * 60)

    # Assert exact replication of claims
    assert round(acc, 4) == round(claimed_acc, 4), f"Accuracy discrepancy: {acc} vs {claimed_acc}"
    assert round(f1_macro, 4) == round(claimed_f1, 4), f"Macro F1 discrepancy: {f1_macro} vs {claimed_f1}"
    print("  [VERIFIED] Exact claims confirmed on hold-out test set!")

    # 5. Per-Class Precision, Recall, F1 Breakdown
    print(f"\n[Step 5] Detailed Per-Class Breakdown (2,400 Test Samples):")
    class_names = list(label_encoder.classes_)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report)

    # 6. Confusion Matrix & Misclassification Error Count
    print(f"[Step 6] Confusion Matrix Analysis:")
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (Rows = True Class, Cols = Predicted Class):")
    header = f"{'Actual Class':<22}" + "".join([f"{c[:10]:>12}" for c in class_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<22}" + "".join([f"{val:>12d}" for val in row])
        print(row_str)

    total_samples = len(y_test)
    correct_samples = int(np.trace(cm))
    misclassified_samples = total_samples - correct_samples
    print(f"\n  Summary:")
    print(f"    - Total Test Samples:       {total_samples}")
    print(f"    - Correctly Classified:     {correct_samples} ({correct_samples / total_samples * 100:.2f}%)")
    print(f"    - Total Errors / False:     {misclassified_samples} ({misclassified_samples / total_samples * 100:.2f}%)")

    # 7. 5-Fold Stratified Cross Validation on Training Set
    print(f"\n[Step 7] 5-Fold Stratified Cross-Validation on Training Data (9,600 samples):")
    X_train_scaled = scaler.transform(X_train_eng)
    X_train_selected = selector.transform(X_train_scaled)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    
    for fold, score in enumerate(cv_scores, 1):
        print(f"    - Fold {fold} Macro F1: {score:.4f} ({score * 100:.2f}%)")
    print(f"  --> Mean CV Macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    assert cv_scores.mean() > 0.98, "Cross-validation score below expected threshold!"
    print("  [VERIFIED] 5-Fold Cross-Validation confirms high stability across all folds.")

    print("\n" + "=" * 80)
    print("  [SUCCESS] ALL ML CLAIMS VERIFIED. ZERO LEAKAGE CONFIRMED.")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
