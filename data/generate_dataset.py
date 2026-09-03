"""
Vehicle Diagnostic Dataset Generator
Grounded in SAE J1979 / OBD-II standard parameters:
- RPM: PID 0x0C
- Engine Coolant Temperature: PID 0x05 (°C)
- Battery / Control Module Voltage: PID 0x42 (V)
- Fuel Rail Pressure: PID 0x0A (psi)
- Calculated Engine Load: PID 0x04 (%)

Classes:
- Normal: Normal vehicle operating envelope
- Cooling System: Overheating, stuck thermostat, radiator/coolant leak (high temp, load correlation)
- Battery/Electrical: Low voltage / alternator failure or overvoltage regulator fault
- Fuel System: Low fuel pump delivery or high pressure regulator lockup
- Engine Mechanical: Rough idle/misfire, abnormal RPM to load disparity
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_vehicle_fault_dataset(num_samples_per_class: int = 2400, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    records = []

    # 1. Normal Operating Condition
    for _ in range(num_samples_per_class):
        rpm = np.random.normal(loc=2100, scale=600)
        rpm = np.clip(rpm, 750, 4200)
        
        # Temp normally between 82°C and 96°C
        temp = np.random.normal(loc=90, scale=4.0)
        temp = np.clip(temp, 80, 99)
        
        # Battery voltage normally 13.6V to 14.4V with alternator charging
        battery = np.random.normal(loc=14.0, scale=0.3)
        battery = np.clip(battery, 13.4, 14.6)
        
        # Fuel pressure normally 40 - 52 psi
        fuel = np.random.normal(loc=46, scale=3.5)
        fuel = np.clip(fuel, 38, 55)
        
        # Engine load matches rpm driving state
        load = (rpm / 4200) * 45 + np.random.normal(loc=20, scale=6.0)
        load = np.clip(load, 15, 75)
        
        records.append({
            "rpm": round(float(rpm), 1),
            "engine_temperature": round(float(temp), 1),
            "battery_voltage": round(float(battery), 2),
            "fuel_pressure": round(float(fuel), 1),
            "engine_load": round(float(load), 1),
            "fault_type": "Normal"
        })

    # 2. Cooling System Fault (Thermostat failure, radiator fan failure, coolant leak)
    for _ in range(num_samples_per_class):
        rpm = np.random.normal(loc=2600, scale=800)
        rpm = np.clip(rpm, 800, 4800)
        
        # High engine temperature: 106°C to 128°C
        temp = np.random.normal(loc=114, scale=5.5)
        temp = np.clip(temp, 104, 130)
        
        # Battery voltage normal to slightly strained due to radiator fan maxing out
        battery = np.random.normal(loc=13.6, scale=0.4)
        battery = np.clip(battery, 12.8, 14.3)
        
        # Fuel pressure normal
        fuel = np.random.normal(loc=45, scale=4.0)
        fuel = np.clip(fuel, 36, 56)
        
        # Engine load runs higher due to thermal throttling and AC/cooling strain
        load = np.random.normal(loc=68, scale=12.0)
        load = np.clip(load, 40, 96)
        
        records.append({
            "rpm": round(float(rpm), 1),
            "engine_temperature": round(float(temp), 1),
            "battery_voltage": round(float(battery), 2),
            "fuel_pressure": round(float(fuel), 1),
            "engine_load": round(float(load), 1),
            "fault_type": "Cooling System"
        })

    # 3. Battery/Electrical Fault (Alternator diode failure, battery degradation, parasitic drain)
    for _ in range(num_samples_per_class):
        rpm = np.random.normal(loc=1900, scale=700)
        rpm = np.clip(rpm, 700, 3800)
        
        temp = np.random.normal(loc=89, scale=5.0)
        temp = np.clip(temp, 78, 101)
        
        # Depleted voltage: 10.2V to 11.9V (or extreme regulator spike > 15.5V)
        if np.random.rand() > 0.15:
            # Low voltage (dying battery / alternator discharge)
            battery = np.random.normal(loc=11.4, scale=0.45)
            battery = np.clip(battery, 9.8, 12.1)
        else:
            # Over-voltage regulator failure
            battery = np.random.normal(loc=15.7, scale=0.4)
            battery = np.clip(battery, 15.2, 16.5)
            
        fuel = np.random.normal(loc=44, scale=4.0)
        fuel = np.clip(fuel, 35, 54)
        
        load = np.random.normal(loc=35, scale=10.0)
        load = np.clip(load, 15, 65)
        
        records.append({
            "rpm": round(float(rpm), 1),
            "engine_temperature": round(float(temp), 1),
            "battery_voltage": round(float(battery), 2),
            "fuel_pressure": round(float(fuel), 1),
            "engine_load": round(float(load), 1),
            "fault_type": "Battery/Electrical"
        })

    # 4. Fuel System Fault (Failing fuel pump, clogged fuel filter, regulator pressure drop)
    for _ in range(num_samples_per_class):
        rpm = np.random.normal(loc=2400, scale=850)
        rpm = np.clip(rpm, 800, 4500)
        
        temp = np.random.normal(loc=93, scale=5.5)
        temp = np.clip(temp, 82, 104)
        
        battery = np.random.normal(loc=13.9, scale=0.35)
        battery = np.clip(battery, 13.2, 14.5)
        
        # Abnormally low fuel pressure: 16 to 29 psi (starvation) or rare high 65-75 psi
        if np.random.rand() > 0.12:
            fuel = np.random.normal(loc=23.5, scale=3.5)
            fuel = np.clip(fuel, 14.0, 31.0)
        else:
            fuel = np.random.normal(loc=68.0, scale=4.0)
            fuel = np.clip(fuel, 62.0, 78.0)
            
        # Engine load increases due to ECU attempting to compensate for lean condition
        load = np.random.normal(loc=62, scale=14.0)
        load = np.clip(load, 30, 92)
        
        records.append({
            "rpm": round(float(rpm), 1),
            "engine_temperature": round(float(temp), 1),
            "battery_voltage": round(float(battery), 2),
            "fuel_pressure": round(float(fuel), 1),
            "engine_load": round(float(load), 1),
            "fault_type": "Fuel System"
        })

    # 5. Engine Mechanical Fault (Severe misfire, valve issue, high load at idle, timing discrepancy)
    for _ in range(num_samples_per_class):
        # Abnormal surging or high rpm under erratic load
        if np.random.rand() > 0.5:
            rpm = np.random.normal(loc=3400, scale=600)
            load = np.random.normal(loc=86, scale=7.0) # High load at moderate rpm
        else:
            rpm = np.random.normal(loc=650, scale=80)  # Rough low idle
            load = np.random.normal(loc=75, scale=8.0) # Extremely high load at idle
            
        rpm = np.clip(rpm, 500, 5200)
        load = np.clip(load, 55, 99)
        
        temp = np.random.normal(loc=98, scale=5.0)
        temp = np.clip(temp, 86, 109)
        
        battery = np.random.normal(loc=13.7, scale=0.4)
        battery = np.clip(battery, 12.8, 14.4)
        
        fuel = np.random.normal(loc=44, scale=4.5)
        fuel = np.clip(fuel, 34, 55)
        
        records.append({
            "rpm": round(float(rpm), 1),
            "engine_temperature": round(float(temp), 1),
            "battery_voltage": round(float(battery), 2),
            "fuel_pressure": round(float(fuel), 1),
            "engine_load": round(float(load), 1),
            "fault_type": "Engine Mechanical"
        })

    df = pd.DataFrame(records)
    
    # Introduce small realistic missingness (~0.5%) to demonstrate ML preprocessing handling
    mask = np.random.rand(*df[["rpm", "engine_temperature", "battery_voltage", "fuel_pressure", "engine_load"]].shape) < 0.005
    feature_cols = ["rpm", "engine_temperature", "battery_voltage", "fuel_pressure", "engine_load"]
    for i, col in enumerate(feature_cols):
        df.loc[mask[:, i], col] = np.nan
        
    # Shuffle
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == "__main__":
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_vehicle_fault_dataset(num_samples_per_class=2400)
    out_file = out_dir / "vehicle_fault_dataset.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated {len(df)} samples across {df['fault_type'].nunique()} fault classes.")
    print("Class distribution:")
    print(df['fault_type'].value_counts())
    print("\nMissing values:")
    print(df.isnull().sum())
