import React from 'react';
import { X, Award, CheckCircle, Database, GitBranch, Layers, Cpu, Sparkles, TrendingUp } from 'lucide-react';

export default function ModelEvaluationModal({ isOpen, onClose, modelInfo }) {
  if (!isOpen) return null;

  // Multi-model benchmark data with live fallback from trained artifacts
  const comparisonModels = modelInfo?.comparison?.models || {
    'LightGBM': {
      accuracy: 0.9900,
      f1_macro: 0.9900,
      precision_macro: 0.9900,
      recall_macro: 0.9900,
      best_cv_score: 0.9906,
    },
    'XGBoost': {
      accuracy: 0.9888,
      f1_macro: 0.9888,
      precision_macro: 0.9888,
      recall_macro: 0.9887,
      best_cv_score: 0.9893,
    },
    'Random Forest': {
      accuracy: 0.9871,
      f1_macro: 0.9871,
      precision_macro: 0.9872,
      recall_macro: 0.9871,
      best_cv_score: 0.9893,
    },
    'MLP (Neural Network)': {
      accuracy: 0.9854,
      f1_macro: 0.9854,
      precision_macro: 0.9855,
      recall_macro: 0.9854,
      best_cv_score: 0.9834,
    },
  };

  const championName = modelInfo?.comparison?.champion_model || modelInfo?.model_name || 'LightGBM';

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Award size={24} color="#00e5ff" />
            <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.4rem' }}>
              ML Model Architecture & Multi-Model Benchmark
            </h2>
          </div>
          <button className="btn-close" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        {/* Key Metrics */}
        <div className="metric-cards-row">
          <div className="metric-stat-card">
            <div className="stat-val">{modelInfo ? `${(modelInfo.accuracy * 100).toFixed(2)}%` : '99.00%'}</div>
            <div className="stat-lbl">Champion Accuracy ({championName})</div>
          </div>
          <div className="metric-stat-card">
            <div className="stat-val">{modelInfo ? modelInfo.f1_score.toFixed(4) : '0.9900'}</div>
            <div className="stat-lbl">Macro F1-Score</div>
          </div>
          <div className="metric-stat-card">
            <div className="stat-val">5 Classes</div>
            <div className="stat-lbl">Subsystem Targets</div>
          </div>
          <div className="metric-stat-card">
            <div className="stat-val">12,000</div>
            <div className="stat-lbl">OBD-II Records</div>
          </div>
        </div>

        {/* Multi-Model Benchmark Comparison Table */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '1.05rem', marginBottom: '10px', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TrendingUp size={18} color="#00e5ff" />
            <span>Multi-Model Architecture Comparison</span>
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.84rem', marginBottom: '12px' }}>
            Identical 80/20 stratified split & 3-Fold GridSearchCV across Tree Ensembles (LightGBM, XGBoost, Random Forest) vs. Deep Feedforward Neural Network (MLP).
          </p>

          <div className="benchmark-table-container">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Model Architecture</th>
                  <th>Type</th>
                  <th>Test Accuracy</th>
                  <th>Macro F1</th>
                  <th>Macro Precision</th>
                  <th>Macro Recall</th>
                  <th>3-Fold CV F1</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparisonModels).map(([name, m]) => {
                  const isChampion = name.toLowerCase().includes(championName.toLowerCase()) || name === 'LightGBM';
                  const modelType = name.includes('MLP') ? 'Neural Network' : name.includes('Forest') ? 'Bagging Ensemble' : 'Boosting Ensemble';
                  return (
                    <tr key={name} className={isChampion ? 'champion-row' : ''}>
                      <td>
                        <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                          {name}
                          {isChampion && (
                            <span className="champion-tag">
                              <Sparkles size={10} /> Champion
                            </span>
                          )}
                        </span>
                      </td>
                      <td style={{ color: 'var(--text-secondary)' }}>{modelType}</td>
                      <td>{(m.accuracy * 100).toFixed(2)}%</td>
                      <td style={{ color: isChampion ? 'var(--accent-cyan)' : '#fff' }}>{m.f1_macro.toFixed(4)}</td>
                      <td>{m.precision_macro.toFixed(4)}</td>
                      <td>{m.recall_macro.toFixed(4)}</td>
                      <td>{m.best_cv_score ? m.best_cv_score.toFixed(4) : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Why Tree Models Win on Tabular Data Card */}
        <div style={{ background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.25)', padding: '18px', borderRadius: '12px', marginBottom: '24px' }}>
          <h4 style={{ color: '#10b981', fontSize: '0.92rem', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={16} />
            <span>Architectural Depth: Why Tree Models Win on Tabular Telemetry</span>
          </h4>
          <p style={{ fontSize: '0.83rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
            {modelInfo?.comparison_insight || (
              "OBD-II vehicle telemetry is governed by step-function physical operating boundaries (e.g. ECT > 105°C for cooling overheating, system voltage < 12.0V for battery failure, rail pressure < 28 PSI for fuel starvation). Tree ensembles (LightGBM, XGBoost, Random Forest) find orthogonal axis-aligned splits that isolate physical thresholds directly without requiring smooth activation approximations or excessive scaling sensitivity, whereas MLPs require smooth sigmoid/ReLU hyperplanes to approximate sharp physical boundaries."
            )}
          </p>
        </div>

        {/* Confusion Matrix Viewer */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '1.05rem', marginBottom: '10px', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={18} color="#00e5ff" />
            <span>Trained Champion Confusion Matrix</span>
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.84rem', marginBottom: '14px' }}>
            Generated during the 80/20 stratified validation test. Demonstrates 99.00% exact isolation across all 5 automotive fault categories.
          </p>
          <div style={{ background: '#000', borderRadius: '12px', padding: '12px', textAlign: 'center', border: '1px solid var(--border-color)' }}>
            <img
              src="/api/confusion-matrix"
              alt="Model Confusion Matrix"
              style={{ maxWidth: '100%', height: 'auto', borderRadius: '8px' }}
              onError={(e) => {
                e.target.style.display = 'none';
              }}
            />
          </div>
        </div>

        {/* Pipeline Details */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px' }}>
          <div style={{ background: 'rgba(0,0,0,0.25)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--accent-cyan)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Database size={14} />
              <span>RAW SENSOR TELEMETRY</span>
            </div>
            <ul style={{ listStyle: 'none', fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: '1.8' }}>
              <li>• Engine RPM (PID 0x0C)</li>
              <li>• Coolant Temperature (PID 0x05)</li>
              <li>• Battery / Module Voltage (PID 0x42)</li>
              <li>• Fuel Rail Pressure (PID 0x0A)</li>
              <li>• Calculated Engine Load (PID 0x04)</li>
            </ul>
          </div>

          <div style={{ background: 'rgba(0,0,0,0.25)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#10b981', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <GitBranch size={14} />
              <span>ENGINEERED FEATURES (OBD-II)</span>
            </div>
            <ul style={{ listStyle: 'none', fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: '1.8' }}>
              <li>• Thermal Stress Index ((ECT - 90)/10)</li>
              <li>• Dynamic Power Demand (RPM * Load)</li>
              <li>• Voltage-to-Fuel Ratio</li>
              <li>• Thermal-Load Cross Interaction</li>
              <li>• RPM-to-Load Discrepancy (Misfires)</li>
            </ul>
          </div>
        </div>

        {/* Interview Highlights Box */}
        <div style={{ background: 'rgba(0, 229, 255, 0.05)', border: '1px solid rgba(0, 229, 255, 0.2)', padding: '16px', borderRadius: '12px' }}>
          <h4 style={{ color: 'var(--accent-cyan)', fontSize: '0.9rem', marginBottom: '6px' }}>
            Why Multiclass Classification (vs Binary "Will It Fail?")
          </h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
            Binary models alert operators to generic failure but lack actionability. Multiclass diagnostic models pinpoint the specific subsystem (e.g. Cooling vs. Electrical vs. Fuel), allowing automated dispatch of exact replacement parts, targeted mechanic inspections, and drastically reduced repair downtime.
          </p>
        </div>
      </div>
    </div>
  );
}
