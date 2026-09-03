"""
SQLAlchemy Data Models for Vehicle Fault Classifier
Stores diagnostic scan logs, telemetry readings, and inference history.
"""

from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, JSON
from .database import Base


class VehicleDiagnosticLog(Base):
    __tablename__ = "vehicle_diagnostic_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Input Telemetry Readings
    rpm = Column(Float, nullable=False)
    engine_temperature = Column(Float, nullable=False)
    battery_voltage = Column(Float, nullable=False)
    fuel_pressure = Column(Float, nullable=False)
    engine_load = Column(Float, nullable=False)

    # Output Diagnostic Prediction
    predicted_fault = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    confidence_percentage = Column(Float, nullable=False)
    severity = Column(String(50), nullable=False)
    diagnostic_code = Column(String(50), nullable=False)
    recommendation = Column(Text, nullable=True)
    probabilities = Column(JSON, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "telemetry": {
                "rpm": self.rpm,
                "engine_temperature": self.engine_temperature,
                "battery_voltage": self.battery_voltage,
                "fuel_pressure": self.fuel_pressure,
                "engine_load": self.engine_load,
            },
            "predicted_fault": self.predicted_fault,
            "confidence": self.confidence,
            "confidence_percentage": self.confidence_percentage,
            "severity": self.severity,
            "diagnostic_code": self.diagnostic_code,
            "recommendation": self.recommendation,
            "probabilities": self.probabilities,
        }
