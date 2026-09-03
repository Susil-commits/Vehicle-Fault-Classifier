"""
Vehicle Fault Classifier - Training Pipeline
End-to-end ML pipeline with:
- Data ingestion & schema validation
- Missing value imputation (fitted strictly on training set)
- Domain-specific feature engineering (thermal stress, power demand, etc.)
- Standardization & Feature Selection
- Multi-model training: Random Forest vs XGBoost
- Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix Heatmaps
- Model serialization and metadata export for production serving
"""

import json
import os
import pickle
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-specific automotive telemetry feature engineering.
    Grounded in OBD-II diagnostic principles.
    """
    df_feat = df.copy()
    
    # 1. Thermal Stress Index: Deviation from optimal 90°C operating window
    df_feat["thermal_stress"] = (df_feat["engine_temperature"] - 90.0) / 10.0
    
    # 2. Power Demand Factor: Dynamic engine workload (RPM * Load / 1000)
    df_feat["power_demand"] = (df_feat["rpm"] * df_feat["engine_load"]) / 1000.0
    
    # 3. Voltage to Fuel Pressure Ratio: Cross-subsystem electrical-to-hydraulic ratio
    df_feat["voltage_fuel_ratio"] = df_feat["battery_voltage"] / (df_feat["fuel_pressure"] + 1e-5)
    
    # 4. Thermal-Load Interaction: High temperatures under high engine loads
    df_feat["temp_load_interaction"] = (df_feat["engine_temperature"] * df_feat["engine_load"]) / 100.0
    
    # 5. RPM to Load Discrepancy: Detects surging or abnormal load at idle
    expected_load = (df_feat["rpm"] / 4200.0) * 50.0
    df_feat["rpm_load_discrepancy"] = np.abs(df_feat["engine_load"] - expected_load)
    
    return df_feat


def run_training(
    data_path: str = "data/raw/vehicle_fault_dataset.csv",
    model_dir: str = "ml/model",
    eval_dir: str = "ml/evaluation",
    random_state: int = 42,
):
    print("=" * 70)
    print("[*] VEHICLE FAULT CLASSIFIER - ML TRAINING PIPELINE")
    print("=" * 70)

    model_path = Path(model_dir)
    eval_path = Path(eval_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    eval_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print(f"\n[1/7] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py first.")

    df = pd.read_csv(data_path)
    print(f"  Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Target Classes Found: {list(df['fault_type'].unique())}")
    print("\n  Class Distribution:")
    for cls, count in df["fault_type"].value_counts().items():
        print(f"    - {cls:22}: {count} ({count/len(df)*100:.1f}%)")

    # 2. Check and Document Missing Values
    print("\n[2/7] Checking missing values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"    - {col}: {count} missing values ({count/len(df)*100:.2f}%)")

    raw_feature_cols = ["rpm", "engine_temperature", "battery_voltage", "fuel_pressure", "engine_load"]
    X_raw = df[raw_feature_cols].copy()
    y_raw = df["fault_type"].copy()

    # 3. Label Encoding for Multi-Class Target
    print("\n[3/7] Encoding target fault categories:")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    class_mapping = {int(i): str(cls) for i, cls in enumerate(label_encoder.classes_)}
    print(f"  Class Index Mapping: {class_mapping}")

    # 4. Strict Featurization Ordering: Train/Test Split BEFORE fitting any transformers!
    print("\n[4/7] Performing Stratified Train/Test Split (80% Train, 20% Test)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_encoded,
        test_size=0.20,
        random_state=random_state,
        stratify=y_encoded,
    )
    print(f"  Training samples: {X_train_raw.shape[0]}")
    print(f"  Testing samples:  {X_test_raw.shape[0]}")

    # Impute missing values using median learned on train
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=raw_feature_cols, index=X_train_raw.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=raw_feature_cols, index=X_test_raw.index)

    # Feature Engineering
    print("  Applying domain feature engineering...")
    X_train_eng = engineer_features(X_train_imp)
    X_test_eng = engineer_features(X_test_imp)
    engineered_feature_cols = list(X_train_eng.columns)
    print(f"  Total features after engineering: {len(engineered_feature_cols)} -> {engineered_feature_cols}")

    # Feature Scaling
    print("  Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_eng)
    X_test_scaled = scaler.transform(X_test_eng)

    # Feature Selection (Select top k features)
    k_features = min(8, len(engineered_feature_cols))
    print(f"  Selecting top {k_features} features using SelectKBest (ANOVA F-score)...")
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    selected_mask = selector.get_support()
    selected_features = [col for col, sel in zip(engineered_feature_cols, selected_mask) if sel]
    print(f"  Selected Features ({len(selected_features)}): {selected_features}")

    # 5. Model Training & Comparison
    print("\n[5/7] Training Models: Random Forest vs XGBoost...")
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"  --> Training {name}...")
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "predictions": y_pred.tolist(),
        }
        fitted_models[name] = model

        print(f"      Accuracy:    {acc * 100:.2f}%")
        print(f"      Macro F1:    {f1_macro:.4f}")
        print(f"      Weighted F1: {f1_weighted:.4f}")

    # Pick the champion model
    best_model_name = max(results.keys(), key=lambda k: results[k]["f1_macro"])
    best_model = fitted_models[best_model_name]
    print(f"\n[CHAMPION] Model Selected: {best_model_name} (F1 = {results[best_model_name]['f1_macro']:.4f})")

    # 6. Detailed Evaluation & Visualizations
    print("\n[6/7] Generating Evaluation Reports and Confusion Matrices...")
    y_best_pred = np.array(results[best_model_name]["predictions"])
    target_names = [label_encoder.classes_[i] for i in range(len(label_encoder.classes_))]

    # Text report
    report_str = classification_report(y_test, y_best_pred, target_names=target_names, digits=4)
    print("\nClassification Report (Champion Model):")
    print(report_str)
    with open(eval_path / "classification_report.txt", "w") as f:
        f.write(f"Vehicle Fault Classifier - Evaluation Report\nModel: {best_model_name}\n\n")
        f.write(report_str)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_best_pred)
    cm_norm = confusion_matrix(y_test, y_best_pred, normalize="true")

    # Plot Raw CM
    plt.figure(figsize=(9, 7), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True,
    )
    plt.title(f"Confusion Matrix - {best_model_name}\nVehicle Fault Classifier", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Fault Type", fontsize=12, fontweight="semibold")
    plt.ylabel("Actual Ground Truth", fontsize=12, fontweight="semibold")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(eval_path / "confusion_matrix.png")
    plt.close()

    # Plot Normalized CM (Percentages)
    plt.figure(figsize=(9, 7), dpi=300)
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True,
    )
    plt.title(f"Normalized Confusion Matrix (%) - {best_model_name}", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Fault Type", fontsize=12, fontweight="semibold")
    plt.ylabel("Actual Ground Truth", fontsize=12, fontweight="semibold")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(eval_path / "confusion_matrix_norm.png")
    plt.close()

    # Feature Importance Plot
    plt.figure(figsize=(10, 6), dpi=300)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        ranked_features = [selected_features[i] for i in indices]
        ranked_scores = importances[indices]
        sns.barplot(x=ranked_scores, y=ranked_features, palette="viridis")
        plt.title(f"Feature Importances ({best_model_name})", fontsize=14, fontweight="bold")
        plt.xlabel("Relative Importance Score", fontsize=12)
        plt.ylabel("Telemetry Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(eval_path / "feature_importance.png")
        plt.close()

    # Export comparison metrics to JSON
    summary_metrics = {
        "champion_model": best_model_name,
        "classes": target_names,
        "raw_features": raw_feature_cols,
        "engineered_features": engineered_feature_cols,
        "selected_features": selected_features,
        "models": {
            name: {
                "accuracy": results[name]["accuracy"],
                "precision_macro": results[name]["precision_macro"],
                "recall_macro": results[name]["recall_macro"],
                "f1_macro": results[name]["f1_macro"],
                "f1_weighted": results[name]["f1_weighted"],
            }
            for name in results
        },
    }
    with open(eval_path / "model_comparison.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    # 7. Persist Artifacts for Production Serving
    print("\n[7/7] Serializing Production Artifacts into ml/model/...")
    with open(model_path / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(model_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(model_path / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    with open(model_path / "feature_selector.pkl", "wb") as f:
        pickle.dump(selector, f)
    with open(model_path / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open(model_path / "rf_model.pkl", "wb") as f:
        pickle.dump(fitted_models["Random Forest"], f)
    with open(model_path / "xgb_model.pkl", "wb") as f:
        pickle.dump(fitted_models["XGBoost"], f)

    # Save model metadata
    model_metadata = {
        "model_name": best_model_name,
        "classes": target_names,
        "raw_feature_cols": raw_feature_cols,
        "engineered_feature_cols": engineered_feature_cols,
        "selected_features": selected_features,
        "test_accuracy": results[best_model_name]["accuracy"],
        "test_f1_score": results[best_model_name]["f1_macro"],
        "version": "1.0.0",
    }
    with open(model_path / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)

    print(f"\nArtifacts successfully exported:")
    print(f"  - {model_path / 'best_model.pkl'}")
    print(f"  - {model_path / 'scaler.pkl'}")
    print(f"  - {model_path / 'imputer.pkl'}")
    print(f"  - {model_path / 'feature_selector.pkl'}")
    print(f"  - {model_path / 'label_encoder.pkl'}")
    print(f"  - {model_path / 'model_metadata.json'}")
    print(f"  - {eval_path / 'confusion_matrix.png'}")
    print(f"  - {eval_path / 'feature_importance.png'}")
    print("\n Training Pipeline Complete!")


if __name__ == "__main__":
    run_training()
