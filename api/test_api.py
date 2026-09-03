"""
Unit & Integration Tests for Vehicle Fault Classifier API
Tests health, schema validation, fault inference, and model metrics.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert data["accuracy"] > 0.95
    assert len(data["classes"]) == 5


def test_user_spec_classification():
    """Test the exact telemetry input specified in the user prompt."""
    payload = {
        "rpm": 3200,
        "engine_temperature": 110,
        "battery_voltage": 11.6,
        "fuel_pressure": 24,
        "engine_load": 82,
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_fault" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert len(data["probabilities"]) == 5
    assert data["severity"] in ["Normal", "Caution", "Warning", "Critical"]
    print(f"\n[Test Result] Input: {payload} -> Predicted: {data['predicted_fault']} (Confidence: {data['confidence_percentage']}%)")


def test_normal_classification():
    payload = {
        "rpm": 2100,
        "engine_temperature": 90.0,
        "battery_voltage": 14.1,
        "fuel_pressure": 46.0,
        "engine_load": 35.0,
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_fault"] == "Normal"
    assert data["confidence"] > 0.80


def test_electrical_fault_classification():
    payload = {
        "rpm": 1800,
        "engine_temperature": 88.0,
        "battery_voltage": 11.2,
        "fuel_pressure": 44.0,
        "engine_load": 30.0,
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_fault"] == "Battery/Electrical"
    assert data["confidence"] > 0.85


def test_invalid_input_validation():
    # Out of range RPM (-100) and engine load (150%)
    invalid_payload = {
        "rpm": -100,
        "engine_temperature": 110,
        "battery_voltage": 12.0,
        "fuel_pressure": 40.0,
        "engine_load": 150.0,
    }
    response = client.post("/classify", json=invalid_payload)
    assert response.status_code == 422  # Pydantic validation error


def test_samples_endpoint():
    response = client.get("/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 5
