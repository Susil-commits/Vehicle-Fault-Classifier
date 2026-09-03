import React from 'react';
import { Gauge, Thermometer, Zap, Droplet, Percent } from 'lucide-react';

export default function LiveGauges({ telemetry }) {
  const gauges = [
    {
      label: 'Engine RPM',
      value: telemetry.rpm,
      unit: 'RPM',
      icon: <Gauge size={16} color="#00e5ff" />,
      color: telemetry.rpm > 5500 ? '#f43f5e' : '#00e5ff',
    },
    {
      label: 'Coolant Temp',
      value: telemetry.engine_temperature,
      unit: '°C',
      icon: <Thermometer size={16} color={telemetry.engine_temperature > 105 ? '#f43f5e' : '#10b981'} />,
      color: telemetry.engine_temperature > 105 ? '#f43f5e' : '#10b981',
    },
    {
      label: 'Battery Voltage',
      value: Number(telemetry.battery_voltage).toFixed(1),
      unit: 'V',
      icon: <Zap size={16} color={telemetry.battery_voltage < 12.0 ? '#f59e0b' : '#38bdf8'} />,
      color: telemetry.battery_voltage < 12.0 ? '#f59e0b' : '#38bdf8',
    },
    {
      label: 'Fuel Pressure',
      value: telemetry.fuel_pressure,
      unit: 'PSI',
      icon: <Droplet size={16} color={telemetry.fuel_pressure < 30 ? '#a855f7' : '#00e5ff'} />,
      color: telemetry.fuel_pressure < 30 ? '#a855f7' : '#00e5ff',
    },
    {
      label: 'Engine Load',
      value: telemetry.engine_load,
      unit: '%',
      icon: <Percent size={16} color={telemetry.engine_load > 80 ? '#f59e0b' : '#10b981'} />,
      color: telemetry.engine_load > 80 ? '#f59e0b' : '#10b981',
    },
  ];

  return (
    <div className="live-gauges-grid">
      {gauges.map((g, idx) => (
        <div key={idx} className="mini-gauge-card">
          <div className="mini-gauge-label" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
            {g.icon}
            <span>{g.label}</span>
          </div>
          <div className="mini-gauge-value" style={{ color: g.color }}>
            {g.value}
            <span className="mini-gauge-unit">{g.unit}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
