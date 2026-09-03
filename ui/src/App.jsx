import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import LiveGauges from './components/LiveGauges';
import DiagnosticForm from './components/DiagnosticForm';
import ResultCard from './components/ResultCard';
import HistorySidebar from './components/HistorySidebar';
import ModelEvaluationModal from './components/ModelEvaluationModal';

// Initial state matches user's exact specification
const INITIAL_TELEMETRY = {
  rpm: 3200,
  engine_temperature: 110,
  battery_voltage: 11.6,
  fuel_pressure: 24,
  engine_load: 82,
};

export default function App() {
  const [telemetry, setTelemetry] = useState(INITIAL_TELEMETRY);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [presets, setPresets] = useState([]);
  const [activePresetId, setActivePresetId] = useState('sample-user-spec');
  const [history, setHistory] = useState([]);
  const [isEvaluationOpen, setIsEvaluationOpen] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Fetch health, presets, model info, and Supabase history on mount
  useEffect(() => {
    fetchMetadata();
    fetchHistory();
    // Run initial classification on mount for the user example
    runInference(INITIAL_TELEMETRY);
  }, []);

  const fetchMetadata = async () => {
    try {
      const [healthRes, samplesRes, modelInfoRes] = await Promise.all([
        fetch('/api/health'),
        fetch('/api/samples'),
        fetch('/api/model-info'),
      ]);

      if (healthRes.ok) {
        const healthData = await healthRes.json();
        setHealth(healthData);
      }

      if (samplesRes.ok) {
        const samplesData = await samplesRes.json();
        setPresets(samplesData);
      }

      if (modelInfoRes.ok) {
        const modelData = await modelInfoRes.json();
        setModelInfo(modelData);
      }
    } catch (err) {
      console.warn('API backend not reachable at /api. Using fallback offline mock.', err);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/history');
      if (res.ok) {
        const logs = await res.json();
        setHistory(logs);
      }
    } catch (err) {
      console.warn('Could not load history from Supabase:', err);
    }
  };

  const handleTelemetryChange = (key, value) => {
    setTelemetry((prev) => ({
      ...prev,
      [key]: value,
    }));
    setActivePresetId(null);
  };

  const handleReset = () => {
    setTelemetry(INITIAL_TELEMETRY);
    setActivePresetId('sample-user-spec');
    runInference(INITIAL_TELEMETRY);
  };

  const handleSelectPreset = (preset) => {
    setTelemetry(preset.telemetry);
    setActivePresetId(preset.id);
    runInference(preset.telemetry);
  };

  const runInference = async (telemetryData) => {
    setLoading(true);
    setErrorMsg(null);

    try {
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(telemetryData),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);

      // Refresh persistent history from Supabase
      fetchHistory();
    } catch (err) {
      console.error('Classification error:', err);
      setErrorMsg('Failed to reach diagnosis engine. Make sure FastAPI server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleClearHistory = async () => {
    try {
      const res = await fetch('/api/history', { method: 'DELETE' });
      if (res.ok) {
        setHistory([]);
      }
    } catch (err) {
      console.error('Error clearing history:', err);
    }
  };

  return (
    <div className="app-container">
      {/* Top Navigation */}
      <Navbar
        health={health}
        onOpenEvaluation={() => setIsEvaluationOpen(true)}
      />

      {/* Error alert if any */}
      {errorMsg && (
        <div
          style={{
            background: 'rgba(244, 63, 94, 0.15)',
            border: '1px solid var(--accent-rose)',
            color: '#fb7185',
            padding: '12px 20px',
            borderRadius: '12px',
            marginBottom: '20px',
            fontSize: '0.85rem',
          }}
        >
          {errorMsg}
        </div>
      )}

      {/* Live Visual Telemetry Header Gauges */}
      <LiveGauges telemetry={telemetry} />

      {/* Main Grid: Form Inputs on Left, Diagnostic Results on Right */}
      <div className="dashboard-grid">
        <DiagnosticForm
          telemetry={telemetry}
          onChange={handleTelemetryChange}
          onAnalyze={() => runInference(telemetry)}
          onReset={handleReset}
          loading={loading}
          presets={presets}
          activePresetId={activePresetId}
          onSelectPreset={handleSelectPreset}
        />

        <div className="results-panel">
          <ResultCard result={result} loading={loading} />
          <HistorySidebar
            history={history}
            onSelectHistory={(item) => {
              if (item.telemetry) {
                setTelemetry(item.telemetry);
              } else if (item.telemetry_received) {
                setTelemetry(item.telemetry_received);
              }
              setResult(item);
            }}
            onClearHistory={handleClearHistory}
          />
        </div>
      </div>

      {/* Model Evaluation & Confusion Matrix Modal */}
      <ModelEvaluationModal
        isOpen={isEvaluationOpen}
        onClose={() => setIsEvaluationOpen(false)}
        modelInfo={modelInfo}
      />
    </div>
  );
}
