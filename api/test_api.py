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


def test_cors_preflight_headers():
    """Verify CORS preflight headers comply with W3C specification (wildcard without credentials)."""
    response = client.options(
        "/classify",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"
    assert response.headers.get("access-control-allow-credentials") is None


def test_delete_history_auth_enforcement():
    """Verify DELETE /history requires valid X-API-Key header."""
    # 1. Unauthenticated request should return 401 Unauthorized
    unauth_response = client.delete("/history")
    assert unauth_response.status_code == 401
    assert "Invalid or missing API key" in unauth_response.json()["detail"]

    # 2. Invalid key should return 401 Unauthorized
    invalid_response = client.delete("/history", headers={"X-API-Key": "wrong-secret-key"})
    assert invalid_response.status_code == 401

    # 3. Valid key should succeed with 200 OK
    valid_response = client.delete("/history", headers={"X-API-Key": "vfc-admin-secret-key"})
    assert valid_response.status_code == 200
    data = valid_response.json()
    assert data["status"] == "cleared"
    assert "deleted_count" in data


def test_model_info_endpoint():
    """Verify /model-info returns architecture, metrics, and multi-model benchmark."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "accuracy" in data
    assert data["accuracy"] > 0.95
    assert "classes" in data
    assert "selected_features" in data

