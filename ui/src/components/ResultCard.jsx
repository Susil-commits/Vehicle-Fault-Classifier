import React from 'react';
import { AlertTriangle, CheckCircle, AlertOctagon, Wrench, ShieldAlert } from 'lucide-react';

export default function ResultCard({ result, loading }) {
  if (!result) {
    return (
      <div className="glass-panel result-card" style={{ textAlign: 'center', padding: '60px 24px' }}>
        <div style={{ opacity: 0.4, marginBottom: '16px' }}>
          <ShieldAlert size={54} color="#00e5ff" style={{ margin: '0 auto' }} />
        </div>
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', marginBottom: '8px' }}>
          Diagnostic Standby
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', maxWidth: '360px', margin: '0 auto' }}>
          Adjust sensor telemetry or pick a preset scenario, then click{' '}
          <strong style={{ color: 'var(--accent-cyan)' }}>ANALYZE</strong> to classify likely vehicle faults.
        </p>
      </div>
    );
  }

  const isNormal = result.predicted_fault === 'Normal';
  const isCritical = result.severity === 'Critical';
  const isWarning = result.severity === 'Warning';

  const badgeClass = isNormal ? 'normal' : isCritical ? 'critical' : 'warning';
  const badgeIcon = isNormal ? (
    <CheckCircle size={14} />
  ) : isCritical ? (
    <AlertOctagon size={14} />
  ) : (
    <AlertTriangle size={14} />
  );

  return (
    <div className="glass-panel result-card">
      <div className="result-header">
        <span className="predicted-title-label">Root Cause Diagnostic Output</span>
        <div className={`fault-badge ${badgeClass}`}>
          {badgeIcon}
          <span>{result.severity} Condition</span>
        </div>
      </div>

      <div className="predicted-fault-display">
        <div className="predicted-title-label">Possible Fault:</div>
        <div className="predicted-fault-name" id="predicted-fault-label">
          {result.predicted_fault}
        </div>
        <div className="fault-dtc-code">
          <span>DTC: {result.diagnostic_code}</span>
        </div>
      </div>

      {/* Animated Confidence Box */}
      <div className="confidence-box">
        <div className="confidence-header">
          <span className="confidence-label">Diagnostic Confidence:</span>
          <span className="confidence-value" id="confidence-value-label">
            {result.confidence_percentage}%
          </span>
        </div>
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{
              width: `${result.confidence_percentage}%`,
              background: isNormal
                ? 'linear-gradient(90deg, #10b981 0%, #059669 100%)'
                : isCritical
                ? 'linear-gradient(90deg, #f43f5e 0%, #e11d48 100%)'
                : 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)',
              boxShadow: isNormal
                ? '0 0 12px rgba(16, 185, 129, 0.4)'
                : isCritical
                ? '0 0 12px rgba(244, 63, 94, 0.4)'
                : '0 0 12px rgba(245, 158, 11, 0.4)',
            }}
          ></div>
        </div>
      </div>

      {/* Technician Action Recommendation */}
      <div className={`recommendation-box ${badgeClass}`}>
        <div className="rec-title">
          <Wrench size={16} />
          <span>Technician Recommendation:</span>
        </div>
        <p className="rec-text">{result.recommendation}</p>
      </div>

      {/* Multi-Class Probabilities Breakdown */}
      {result.probabilities && (
        <div className="probabilities-section">
          <div className="prob-title">Subsystem Probability Distribution</div>
          <div className="prob-list">
            {Object.entries(result.probabilities)
              .sort(([, a], [, b]) => b - a)
              .map(([cls, prob]) => {
                const pct = (prob * 100).toFixed(1);
                const isChampion = cls === result.predicted_fault;
                return (
                  <div key={cls} className="prob-item">
                    <div className="prob-meta">
                      <span style={{ color: isChampion ? '#fff' : 'var(--text-secondary)' }}>
                        {cls}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: isChampion ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
                        {pct}%
                      </span>
                    </div>
                    <div className="prob-bar-track">
                      <div
                        className={`prob-bar-fill ${isChampion ? 'active' : ''}`}
                        style={{ width: `${pct}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}
