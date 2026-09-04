"""
FastAPI REST API for Vehicle Fault Classifier
Exposes:
- POST /classify   : Real-time telemetry fault prediction & automatic Supabase DB logging
- GET  /health     : API, ML model, and Supabase PostgreSQL health status
- GET  /history    : Retrieves persistent diagnostic history from Supabase
- DELETE /history  : Clears stored diagnostic logs
- GET  /model-info : Preprocessing & ML evaluation metrics
- GET  /samples    : Diagnostic sample presets for 1-click testing
- GET  /confusion-matrix : Serves the generated evaluation heatmap
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import VehicleDiagnosticLog
from .predictor import VehicleFaultPredictor
from .schemas import (
    ClassificationOutput,
    HealthResponse,
    ModelInfoResponse,
    PresetSample,
    VehicleTelemetryInput,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist in Supabase PostgreSQL
    try:
        Base.metadata.create_all(bind=engine)
        print("[DB] Tables verified/created in Supabase PostgreSQL.")
    except Exception as e:
        print(f"[DB ERROR] Startup table creation error: {e}")
    yield


app = FastAPI(
    title="Vehicle Fault Classifier API",
    description="OBD-II Telemetry Multiclass Fault Diagnosis API with Supabase PostgreSQL Persistence",
    version="1.1.0",
    lifespan=lifespan,
)

# Enable CORS for frontend UI (allow_credentials=False complies with W3C CORS spec when using allow_origins=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor singleton
predictor = VehicleFaultPredictor.get_instance()


def require_admin_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Lightweight administrative API key check for destructive operations (DELETE /history).
    Converts documented known limitation into an active access control.
    """
    expected_key = os.getenv("VFC_API_KEY", "vfc-admin-secret-key")
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide a valid 'X-API-Key' header.",
        )
    return x_api_key


@app.get("/health", tags=["Diagnostics"])
def get_health(db: Session = Depends(get_db)):
    """Health check endpoint confirming API, ML model, and Supabase PostgreSQL readiness."""
    meta = predictor.metadata
    db_status = "Supabase PostgreSQL (Connected)"
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1")).scalar()
    except Exception as e:
        db_status = f"Disconnected ({str(e)})"

    return {
        "status": "healthy",
        "database": db_status,
        "model": meta.get("model_name", "Random Forest Classifier"),
        "version": meta.get("version", "1.1.0"),
        "accuracy": meta.get("test_accuracy", 0.9875),
        "f1_score": meta.get("test_f1_score", 0.9875),
        "classes": meta.get("classes", [str(c) for c in predictor.label_encoder.classes_]),
    }


@app.post("/classify", response_model=ClassificationOutput, tags=["Inference"])
def classify_telemetry(payload: VehicleTelemetryInput, db: Session = Depends(get_db)):
    """
    Submits vehicle telemetry (RPM, Temperature, Battery Voltage, Fuel Pressure, Engine Load),
    predicts the likely subsystem fault category, and logs the scan to Supabase PostgreSQL.
    """
    try:
        telemetry_dict = payload.model_dump()
        result = predictor.predict(telemetry_dict)

        # Log into Supabase PostgreSQL
        try:
            log_entry = VehicleDiagnosticLog(
                rpm=payload.rpm,
                engine_temperature=payload.engine_temperature,
                battery_voltage=payload.battery_voltage,
                fuel_pressure=payload.fuel_pressure,
                engine_load=payload.engine_load,
                predicted_fault=result["predicted_fault"],
                confidence=result["confidence"],
                confidence_percentage=result["confidence_percentage"],
                severity=result["severity"],
                diagnostic_code=result["diagnostic_code"],
                recommendation=result["recommendation"],
                probabilities=result["probabilities"],
            )
            db.add(log_entry)
            db.commit()
        except Exception as db_err:
            db.rollback()
            print(f"[DB WARNING] Failed to persist scan to database: {db_err}")

        return ClassificationOutput(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.get("/history", tags=["Diagnostics"])
def get_diagnostic_history(limit: int = 20, db: Session = Depends(get_db)):
    """Retrieves recent diagnostic logs stored in Supabase PostgreSQL."""
    try:
        logs = (
            db.query(VehicleDiagnosticLog)
            .order_by(VehicleDiagnosticLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [log.to_dict() for log in logs]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(e)}",
        )


@app.delete("/history", tags=["Diagnostics"])
def clear_diagnostic_history(
    _auth: str = Depends(require_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Clears all stored diagnostic logs in the database.
    Protected by lightweight administrative API Key authentication (X-API-Key header).
    """
    try:
        count = db.query(VehicleDiagnosticLog).delete()
        db.commit()
        return {"status": "cleared", "deleted_count": count}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Diagnostics"])
def get_model_info():
    """Returns detailed architecture, feature selection, evaluation scores, and multi-model benchmark."""
    meta = predictor.metadata
    comparison = None
    comparison_file = Path(__file__).resolve().parent.parent / "ml" / "evaluation" / "model_comparison.json"
    if comparison_file.exists():
        try:
            with open(comparison_file, "r") as f:
                comparison = json.load(f)
        except Exception as e:
            print(f"[Model Info WARNING] Could not load model comparison: {e}")

    insight = (
        "Tree models (XGBoost, LightGBM, Random Forest) outperform neural architectures (MLP) on tabular "
        "OBD-II telemetry because physical failure signatures are governed by orthogonal step-function "
        "threshold boundaries (e.g. ECT > 105°C, Voltage < 12.0V, Rail Pressure < 28 PSI). Tree partitions "
        "isolate these threshold boundaries directly, whereas MLPs require smooth sigmoid/ReLU hyperplanes "
        "and extensive regularization across unnormalized tabular feature interactions."
    )

    return ModelInfoResponse(
        model_name=meta.get("model_name", "XGBoost Classifier"),
        version=meta.get("version", "1.1.0"),
        accuracy=meta.get("test_accuracy", 0.9888),
        f1_score=meta.get("test_f1_score", 0.9888),
        classes=meta.get("classes", [str(c) for c in predictor.label_encoder.classes_]),
        raw_features=meta.get("raw_feature_cols", []),
        engineered_features=meta.get("engineered_feature_cols", []),
        selected_features=meta.get("selected_features", []),
        comparison=comparison,
        comparison_insight=insight,
    )


@app.get("/samples", response_model=List[PresetSample], tags=["Presets"])
def get_sample_presets():
    """Provides curated telemetry presets for rapid demonstration in the UI."""
    return [
        PresetSample(
            id="sample-user-spec",
            title="User Example (Cooling / High Temp)",
            category="Cooling System",
            description="High temperature (110°C) with elevated load (82%) indicative of thermal cooling failure.",
            telemetry=VehicleTelemetryInput(
                rpm=3200,
                engine_temperature=110,
                battery_voltage=11.6,
                fuel_pressure=24,
                engine_load=82,
            ),
        ),
        PresetSample(
            id="sample-normal",
            title="Normal Cruising Condition",
            category="Normal",
            description="Optimal cruising parameters with balanced thermal, electrical, and fuel envelopes.",
            telemetry=VehicleTelemetryInput(
                rpm=2100,
                engine_temperature=90.5,
                battery_voltage=14.1,
                fuel_pressure=46.0,
                engine_load=38.0,
            ),
        ),
        PresetSample(
            id="sample-cooling",
            title="Radiator / Thermostat Overheating",
            category="Cooling System",
            description="Extreme engine coolant temperature (118°C) with elevated thermal load.",
            telemetry=VehicleTelemetryInput(
                rpm=2900,
                engine_temperature=118.0,
                battery_voltage=13.6,
                fuel_pressure=45.0,
                engine_load=75.0,
            ),
        ),
        PresetSample(
            id="sample-battery",
            title="Alternator / Battery Failure",
            category="Battery/Electrical",
            description="Severely depleted charging voltage (11.1V) threatening electrical blackout.",
            telemetry=VehicleTelemetryInput(
                rpm=1800,
                engine_temperature=88.0,
                battery_voltage=11.1,
                fuel_pressure=44.0,
                engine_load=32.0,
            ),
        ),
        PresetSample(
            id="sample-fuel",
            title="Fuel Pump Starvation",
            category="Fuel System",
            description="Severely depressed fuel rail pressure (21.5 psi) causing lean starvation.",
            telemetry=VehicleTelemetryInput(
                rpm=2600,
                engine_temperature=93.0,
                battery_voltage=13.8,
                fuel_pressure=21.5,
                engine_load=66.0,
            ),
        ),
        PresetSample(
            id="sample-engine",
            title="Cylinder Misfire / Mechanical Drag",
            category="Engine Mechanical",
            description="Abnormally high engine load (88%) at low idle RPM (680), showing severe engine drag.",
            telemetry=VehicleTelemetryInput(
                rpm=680,
                engine_temperature=97.0,
                battery_voltage=13.4,
                fuel_pressure=43.0,
                engine_load=88.0,
            ),
        ),
    ]


@app.get("/confusion-matrix", tags=["Diagnostics"])
def get_confusion_matrix():
    """Serves the generated confusion matrix heatmap image."""
    cm_path = Path(__file__).resolve().parent.parent / "ml" / "evaluation" / "confusion_matrix.png"
    if not cm_path.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix image not found")
    return FileResponse(str(cm_path), media_type="image/png")
