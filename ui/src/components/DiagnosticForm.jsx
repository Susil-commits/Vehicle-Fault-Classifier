import React from 'react';
import { Sliders, Play, RotateCcw, Sparkles } from 'lucide-react';

export default function DiagnosticForm({
  telemetry,
  onChange,
  onAnalyze,
  onReset,
  loading,
  presets,
  activePresetId,
  onSelectPreset,
}) {
  const fields = [
    {
      key: 'rpm',
      label: 'Engine Speed (RPM)',
      pid: 'PID 0x0C',
      min: 500,
      max: 6500,
      step: 50,
      unit: 'RPM',
      nominal: '750 - 3500 RPM',
    },
    {
      key: 'engine_temperature',
      label: 'Coolant Temperature',
      pid: 'PID 0x05',
      min: 50,
      max: 135,
      step: 1,
      unit: '°C',
      nominal: '82 - 96 °C',
    },
    {
      key: 'battery_voltage',
      label: 'Battery / Alternator Voltage',
      pid: 'PID 0x42',
      min: 9.5,
      max: 16.5,
      step: 0.1,
      unit: 'V',
      nominal: '13.6 - 14.4 V',
    },
    {
      key: 'fuel_pressure',
      label: 'Fuel Rail Pressure',
      pid: 'PID 0x0A',
      min: 12,
      max: 75,
      step: 1,
      unit: 'PSI',
      nominal: '40 - 52 PSI',
    },
    {
      key: 'engine_load',
      label: 'Calculated Engine Load',
      pid: 'PID 0x04',
      min: 10,
      max: 100,
      step: 1,
      unit: '%',
      nominal: '15 - 65 %',
    },
  ];

  return (
    <div className="glass-panel form-panel">
      <div className="section-title-bar">
        <div className="section-title">
          <Sliders size={20} color="#00e5ff" />
          <span>Vehicle Sensor Telemetry Input</span>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="preset-chip"
          title="Reset telemetry to default standard values"
        >
          <RotateCcw size={14} />
          <span>Reset</span>
        </button>
      </div>

      {/* Preset Diagnostic Scenarios */}
      <div className="presets-container">
        <div className="presets-label">
          <Sparkles size={12} style={{ display: 'inline', marginRight: '6px' }} />
          Select Diagnostic Scenario Preset:
        </div>
        <div className="preset-chips">
          {presets.map((p) => (
            <button
              key={p.id}
              type="button"
              className={`preset-chip ${activePresetId === p.id ? 'active' : ''}`}
              onClick={() => onSelectPreset(p)}
              title={p.description}
            >
              <span>{p.title}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Inputs with synced sliders */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onAnalyze();
        }}
      >
        <div className="telemetry-fields">
          {fields.map((f) => (
            <div key={f.key} className="telemetry-row">
              <div className="telemetry-header">
                <div className="telemetry-name">
                  <span>{f.label}</span>
                  <span className="telemetry-pid">{f.pid}</span>
                </div>
                <div className="telemetry-value-input">
                  <input
                    type="number"
                    id={`input-${f.key}`}
                    className="numeric-input"
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    value={telemetry[f.key]}
                    onChange={(e) => onChange(f.key, parseFloat(e.target.value) || 0)}
                  />
                  <span className="unit-tag">{f.unit}</span>
                </div>
              </div>

              <div className="telemetry-controls">
                <input
                  type="range"
                  id={`slider-${f.key}`}
                  className="slider-input"
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  value={telemetry[f.key]}
                  onChange={(e) => onChange(f.key, parseFloat(e.target.value))}
                />
              </div>

              <div className="range-bounds">
                <span>Min: {f.min} {f.unit}</span>
                <span style={{ color: 'var(--text-secondary)' }}>Nominal: {f.nominal}</span>
                <span>Max: {f.max} {f.unit}</span>
              </div>
            </div>
          ))}
        </div>

        <button
          type="submit"
          id="btn-analyze-diagnostic"
          className="btn-analyze"
          disabled={loading}
        >
          {loading ? (
            <>
              <div className="spinner"></div>
              <span>Processing ML Inference...</span>
            </>
          ) : (
            <>
              <Play size={18} fill="#000" />
              <span>ANALYZE DIAGNOSTIC</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
