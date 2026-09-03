"""
Pydantic Schemas for Vehicle Fault Classifier API
Defines input telemetry schema with realistic bounds and output diagnostic results.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class VehicleTelemetryInput(BaseModel):
    rpm: float = Field(
        ...,
        ge=0,
        le=9000,
        description="Engine Revolutions Per Minute (RPM)",
        examples=[3200.0],
    )
    engine_temperature: float = Field(
        ...,
        ge=-40,
        le=150,
        description="Engine Coolant Temperature in °C",
        examples=[110.0],
    )
    battery_voltage: float = Field(
        ...,
        ge=5.0,
        le=20.0,
        description="Control Module / Battery Voltage in Volts",
        examples=[11.6],
    )
    fuel_pressure: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Fuel Rail Pressure in PSI",
        examples=[24.0],
    )
    engine_load: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Calculated Engine Load in %",
        examples=[82.0],
    )


class ClassificationOutput(BaseModel):
    predicted_fault: str = Field(..., description="Likely vehicle subsystem fault identified")
    confidence: float = Field(..., description="Prediction probability confidence between 0 and 1")
    confidence_percentage: float = Field(..., description="Confidence formatted as percentage (0-100)")
    severity: str = Field(..., description="Severity level: Normal, Caution, Warning, or Critical")
    diagnostic_code: str = Field(..., description="Mapped SAE / OBD-II Diagnostic Trouble Code")
    sae_definition: Optional[str] = Field(None, description="Official SAE J2012 DTC Title")
    subsystem: Optional[str] = Field(None, description="Automotive powertrain subsystem")
    recommendation: str = Field(..., description="Recommended diagnostic and technician inspection step")
    probabilities: Dict[str, float] = Field(..., description="Confidence distribution across all fault classes")
    telemetry_received: Dict[str, float] = Field(..., description="Echoed input telemetry")


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    accuracy: float
    f1_score: float
    classes: List[str]


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    accuracy: float
    f1_score: float
    classes: List[str]
    raw_features: List[str]
    engineered_features: List[str]
    selected_features: List[str]


class PresetSample(BaseModel):
    id: str
    title: str
    category: str
    description: str
    telemetry: VehicleTelemetryInput
