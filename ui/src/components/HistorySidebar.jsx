import React from 'react';
import { History, Database, Trash2, ArrowRight } from 'lucide-react';

export default function HistorySidebar({ history, onSelectHistory, onClearHistory }) {
  return (
    <div className="glass-panel" style={{ padding: '20px', marginTop: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Database size={16} color="#00e5ff" />
          <span style={{ fontSize: '0.82rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
            Supabase Diagnostic Logs ({history.length})
          </span>
        </div>

        {history.length > 0 && (
          <button
            type="button"
            onClick={onClearHistory}
            className="preset-chip"
            style={{ padding: '4px 8px', fontSize: '0.72rem' }}
            title="Clear all stored logs in Supabase database"
          >
            <Trash2 size={12} />
            <span>Clear</span>
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', textAlign: 'center', padding: '16px 0' }}>
          No diagnostic logs recorded yet. Run an analysis to log to Supabase PostgreSQL.
        </div>
      ) : (
        <div className="history-list">
          {history.slice(0, 8).map((item, index) => {
            const isNormal = item.predicted_fault === 'Normal';
            const isCritical = item.severity === 'Critical';
            const dotColor = isNormal ? '#10b981' : isCritical ? '#f43f5e' : '#f59e0b';
            
            // Format time
            const formattedTime = item.created_at
              ? new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
              : `Scan #${item.id || index + 1}`;

            return (
              <div
                key={item.id || index}
                className="history-item"
                style={{ cursor: 'pointer' }}
                onClick={() => onSelectHistory(item)}
                title="Click to reload this diagnostic telemetry into the form"
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span
                    style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      backgroundColor: dotColor,
                      display: 'inline-block',
                      flexShrink: 0,
                    }}
                  ></span>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.82rem' }}>{item.predicted_fault}</div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{formattedTime}</div>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>
                    {item.confidence_percentage}%
                  </span>
                  <ArrowRight size={12} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
