"""
Blast Simulator Module
Physics-based simulation of explosive events
"""

import numpy as np
from typing import Dict, Tuple
import logging


class BlastSimulator:
    """
    Simulate blast wave physics for testing.
    
    Based on Kingery-Bulmash equations for blast overpressure
    and Friedlander waveform for pressure-time history.
    """
    
    def __init__(self):
        self.c_sound = 343  # Speed of sound in air (m/s)
        logging.info("Blast simulator initialized")
    
    def simulate_blast(self, 
                      explosive_mass: float,
                      distance: float,
                      sample_rate: int = 1000,
                      duration: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Simulate blast wave from explosion.
        
        Args:
            explosive_mass: TNT equivalent mass in kg
            distance: Distance from explosion in meters
            sample_rate: Sampling rate in Hz
            duration: Duration of simulation in seconds
        
        Returns:
            Dictionary with pressure waveform and derived data
        """
        # Calculate scaled distance (Hopkinson-Cranz scaling)
        z = distance / (explosive_mass ** (1/3))  # m/kg^(1/3)
        
        # Calculate peak overpressure using Kingery-Bulmash
        peak_pressure = self._calculate_peak_overpressure(z)
        
        # Calculate arrival time
        t_arrival = distance / self.c_sound
        
        # Generate pressure-time history
        t = np.linspace(0, duration, int(duration * sample_rate))
        pressure = self._friedlander_waveform(
            t, t_arrival, peak_pressure, explosive_mass, distance
        )
        
        # Calculate impulse (area under curve)
        impulse = np.trapz(np.maximum(pressure, 0), t)
        
        # Generate acceleration from pressure wave
        # F = P * A, a = F/m (assuming chest area ~0.2 m²)
        chest_area = 0.2  # m²
        body_mass = 80  # kg
        acceleration = (pressure * chest_area) / body_mass
        
        return {
            'time': t,
            'pressure': pressure,
            'acceleration': acceleration,
            'peak_pressure': peak_pressure,
            'impulse': impulse,
            'scaled_distance': z,
            't_arrival': t_arrival,
            'distance': distance,
            'tnt_mass': explosive_mass
        }
    
    def _calculate_peak_overpressure(self, z: float) -> float:
        """
        Calculate peak overpressure using Kingery-Bulmash equations.
        
        Args:
            z: Scaled distance (m/kg^(1/3))
        
        Returns:
            Peak overpressure in Pa
        """
        # Simplified Kingery-Bulmash formula
        # Full equation is more complex
        
        if z < 0.1:  # Very close (unrealistic for survivability)
            P = 1e7
        elif z < 0.3:
            P = 1e6 / z**2
        elif z < 1.0:
            P = 8e5 / z**2.5
        elif z < 10.0:
            P = 5e5 / z**3
        else:
            P = 2e5 / z**4
        
        return P
    
    def _friedlander_waveform(self,
                             t: np.ndarray,
                             t_arrival: float,
                             peak_pressure: float,
                             mass: float,
                             distance: float) -> np.ndarray:
        """
        Generate Friedlander waveform for blast pressure.
        
        P(t) = P_max * (1 - (t-t_a)/t_d) * exp(-b * (t-t_a)/t_d)
        
        Args:
            t: Time array
            t_arrival: Arrival time of shock front
            peak_pressure: Peak overpressure
            mass: Explosive mass
            distance: Distance from blast
        
        Returns:
            Pressure waveform
        """
        # Positive phase duration
        t_duration = 0.001 * mass**(1/3) * (1 + (distance / (mass**(1/3)))**0.5)
        
        # Decay constant
        b = 2.0
        
        # Initialize pressure array
        pressure = np.zeros_like(t)
        
        # Positive phase
        mask_positive = (t >= t_arrival) & (t < t_arrival + t_duration)
        t_rel = (t[mask_positive] - t_arrival) / t_duration
        pressure[mask_positive] = peak_pressure * (1 - t_rel) * np.exp(-b * t_rel)
        
        # Negative phase (underpressure)
        t_negative = t_duration * 3  # Negative phase is longer
        mask_negative = (t >= t_arrival + t_duration) & (t < t_arrival + t_duration + t_negative)
        t_rel_neg = (t[mask_negative] - t_arrival - t_duration) / t_negative
        pressure[mask_negative] = -0.3 * peak_pressure * np.exp(-t_rel_neg)
        
        return pressure
    
    def estimate_injury_severity(self, peak_pressure: float, impulse: float) -> Dict:
        """
        Estimate injury severity from blast parameters.
        
        Uses injury criteria based on peak overpressure and impulse.
        
        Args:
            peak_pressure: Peak overpressure (Pa)
            impulse: Pressure impulse (Pa·s)
        
        Returns:
            Dictionary with injury assessment
        """
        # Convert to kPa for comparison
        peak_kpa = peak_pressure / 1000
        
        # Bowen curves for blast injury
        severity = {
            'eardrum_rupture': peak_kpa > 35,
            'lung_damage': peak_kpa > 100,
            'severe_lung_injury': peak_kpa > 200,
            'skull_fracture': peak_kpa > 300,
            'lethality_50': peak_kpa > 500,
            'severity_score': 0.0,
            'primary_injury_risk': 'LOW'
        }
        
        # Calculate overall severity score
        if peak_kpa < 35:
            severity['severity_score'] = 0.1
            severity['primary_injury_risk'] = 'MINIMAL'
        elif peak_kpa < 100:
            severity['severity_score'] = 0.3
            severity['primary_injury_risk'] = 'MODERATE'
        elif peak_kpa < 200:
            severity['severity_score'] = 0.6
            severity['primary_injury_risk'] = 'SEVERE'
        elif peak_kpa < 500:
            severity['severity_score'] = 0.85
            severity['primary_injury_risk'] = 'CRITICAL'
        else:
            severity['severity_score'] = 1.0
            severity['primary_injury_risk'] = 'LETHAL'
        
        # Impulse-based assessment
        if impulse > 2000:  # Pa·s
            severity['severity_score'] = min(1.0, severity['severity_score'] + 0.2)
        
        return severity
    
    def simulate_scenario(self, scenario_type: str) -> Dict:
        """
        Simulate predefined blast scenarios.
        
        Args:
            scenario_type: Type of scenario
                - 'grenade': Hand grenade (0.5 kg TNT)
                - 'ied_small': Small IED (5 kg TNT)
                - 'ied_large': Large IED (50 kg TNT)
                - 'artillery': Artillery shell (10 kg TNT)
                - 'mortar': Mortar round (3 kg TNT)
        
        Returns:
            Blast simulation results
        """
        scenarios = {
            'grenade': {'mass': 0.5, 'distance': 10},
            'ied_small': {'mass': 5, 'distance': 15},
            'ied_large': {'mass': 50, 'distance': 30},
            'artillery': {'mass': 10, 'distance': 20},
            'mortar': {'mass': 3, 'distance': 12}
        }
        
        if scenario_type not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_type}")
        
        params = scenarios[scenario_type]
        result = self.simulate_blast(params['mass'], params['distance'])
        
        # Add injury assessment
        injury = self.estimate_injury_severity(
            result['peak_pressure'],
            result['impulse']
        )
        result['injury_assessment'] = injury
        
        return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    logging.basicConfig(level=logging.INFO)
    simulator = BlastSimulator()
    
    print("="*60)
    print("BLAST SIMULATOR TEST")
    print("="*60)
    
    # Test scenarios
    scenarios = ['grenade', 'ied_small', 'artillery']
    
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 10))
    
    for idx, scenario in enumerate(scenarios):
        print(f"\n{scenario.upper()}:")
        result = simulator.simulate_scenario(scenario)
        
        print(f"  TNT mass: {result['tnt_mass']} kg")
        print(f"  Distance: {result['distance']} m")
        print(f"  Peak pressure: {result['peak_pressure']/1000:.1f} kPa")
        print(f"  Impulse: {result['impulse']:.1f} Pa·s")
        print(f"  Injury risk: {result['injury_assessment']['primary_injury_risk']}")
        print(f"  Severity: {result['injury_assessment']['severity_score']:.2f}")
        
        # Plot
        ax = axes[idx]
        ax.plot(result['time'] * 1000, result['pressure'] / 1000, 'r-', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('Overpressure (kPa)')
        ax.set_title(f"{scenario.upper()} - {result['tnt_mass']} kg TNT at {result['distance']} m")
        ax.grid(True, alpha=0.3)
        
        if idx == len(scenarios) - 1:
            ax.set_xlabel('Time (ms)')
    
    plt.tight_layout()
    plt.savefig('blast_simulation.png', dpi=300)
    print("\n✓ Plot saved to blast_simulation.png")
