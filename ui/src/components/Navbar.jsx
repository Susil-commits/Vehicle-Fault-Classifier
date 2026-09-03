import React from 'react';
import { Activity, Cpu, ShieldCheck, BarChart2 } from 'lucide-react';

export default function Navbar({ health, onOpenEvaluation }) {
  return (
    <header className="app-header glass-panel">
      <div className="brand-wrapper">
        <div className="brand-icon">
          <Activity size={24} />
        </div>
        <div className="brand-text">
          <h1>Vehicle Fault Classifier</h1>
          <p>OBD-II Multiclass Diagnostic Intelligence System</p>
        </div>
      </div>

      <div className="header-status">
        <div className="status-badge" title="Database Connection">
          <span className="status-dot"></span>
          <span>{health?.database || 'Supabase PostgreSQL'}</span>
        </div>

        <div className="status-badge" title="Trained Model Status">
          <Cpu size={14} />
          <span>{health?.model || 'Random Forest'} • {health?.accuracy ? `${(health.accuracy * 100).toFixed(1)}% Acc` : 'Online'}</span>
        </div>

        <button
          className="tab-btn"
          onClick={onOpenEvaluation}
          title="View Model Metrics & Confusion Matrix"
        >
          <BarChart2 size={16} />
          <span>Model Metrics</span>
        </button>
      </div>
    </header>
  );
}
