"""
Scenario Generator
Creates realistic military combat scenarios for testing
"""

import numpy as np
from typing import Dict, List
import random


class ScenarioGenerator:
    """Generate realistic combat scenarios for testing and validation"""
    
    SCENARIOS = {
        'patrol_ambush': {
            'description': 'Patrol encounters IED and small arms fire',
            'duration': 300,  # seconds
            'events': [
                {'type': 'ied_small', 'time': 30, 'soldiers_affected': [1, 2]},
                {'type': 'gunshot', 'time': 32, 'soldiers_affected': [3]},
                {'type': 'gunshot', 'time': 35, 'soldiers_affected': [1]},
            ]
        },
        'artillery_strike': {
            'description': 'Artillery barrage on position',
            'duration': 180,
            'events': [
                {'type': 'artillery', 'time': 10, 'soldiers_affected': [1, 2, 3, 4]},
                {'type': 'artillery', 'time': 25, 'soldiers_affected': [2, 5]},
                {'type': 'artillery', 'time': 45, 'soldiers_affected': [3, 4, 6]},
            ]
        },
        'vehicle_ied': {
            'description': 'Vehicle convoy hits IED',
            'duration': 120,
            'events': [
                {'type': 'ied_large', 'time': 15, 'soldiers_affected': [1, 2, 3, 4, 5]},
                {'type': 'vehicle_crash', 'time': 15.5, 'soldiers_affected': [1, 2, 3]},
            ]
        },
        'sniper_attack': {
            'description': 'Sniper engagement',
            'duration': 600,
            'events': [
                {'type': 'gunshot', 'time': 120, 'soldiers_affected': [3]},
                {'type': 'gunshot', 'time': 245, 'soldiers_affected': [1]},
                {'type': 'gunshot', 'time': 380, 'soldiers_affected': [5]},
            ]
        },
        'building_collapse': {
            'description': 'Building collapse during urban operation',
            'duration': 60,
            'events': [
                {'type': 'blast', 'time': 5, 'soldiers_affected': [1, 2, 3]},
                {'type': 'fall', 'time': 6, 'soldiers_affected': [1, 4]},
                {'type': 'fall', 'time': 8, 'soldiers_affected': [2]},
            ]
        }
    }
    
    def __init__(self, squad_size=8):
        self.squad_size = squad_size
        self.soldiers = [f"GS-{i+1:03d}" for i in range(squad_size)]
    
    def generate_scenario(self, scenario_name: str) -> Dict:
        """
        Generate a complete scenario with all events.
        
        Args:
            scenario_name: Name of scenario from SCENARIOS dict
        
        Returns:
            Complete scenario data with timeline
        """
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.SCENARIOS[scenario_name].copy()
        
        # Add soldier assignments
        timeline = []
        for event in scenario['events']:
            for soldier_idx in event['soldiers_affected']:
                if soldier_idx <= self.squad_size:
                    timeline.append({
                        'time': event['time'],
                        'device_id': self.soldiers[soldier_idx - 1],
                        'event_type': event['type'],
                        'scenario': scenario_name
                    })
        
        # Sort by time
        timeline.sort(key=lambda x: x['time'])
        
        return {
            'name': scenario_name,
            'description': scenario['description'],
            'duration': scenario['duration'],
            'timeline': timeline,
            'total_events': len(timeline)
        }
    
    def generate_random_scenario(self, duration=300, event_rate=0.05) -> Dict:
        """
        Generate random scenario with events distributed over time.
        
        Args:
            duration: Scenario duration in seconds
            event_rate: Average events per second
        
        Returns:
            Random scenario data
        """
        event_types = ['blast', 'gunshot', 'artillery', 'ied_small', 'fall']
        num_events = int(duration * event_rate)
        
        timeline = []
        for _ in range(num_events):
            timeline.append({
                'time': random.uniform(0, duration),
                'device_id': random.choice(self.soldiers),
                'event_type': random.choice(event_types),
                'scenario': 'random'
            })
        
        timeline.sort(key=lambda x: x['time'])
        
        return {
            'name': 'random',
            'description': f'Randomly generated scenario with {num_events} events',
            'duration': duration,
            'timeline': timeline,
            'total_events': len(timeline)
        }
    
    def get_available_scenarios(self) -> List[str]:
        """Get list of available scenarios"""
        return list(self.SCENARIOS.keys())
    
    def print_scenario(self, scenario: Dict):
        """Pretty print scenario details"""
        print("="*60)
        print(f"SCENARIO: {scenario['name'].upper()}")
        print("="*60)
        print(f"Description: {scenario['description']}")
        print(f"Duration: {scenario['duration']} seconds")
        print(f"Total Events: {scenario['total_events']}")
        print("\nTimeline:")
        print("-"*60)
        
        for event in scenario['timeline']:
            print(f"  T+{event['time']:6.1f}s | {event['device_id']} | {event['event_type']}")
        
        print("="*60)


if __name__ == "__main__":
    generator = ScenarioGenerator(squad_size=8)
    
    print("GUARDIAN-SHIELD Scenario Generator\n")
    
    # List available scenarios
    print("Available scenarios:")
    for name in generator.get_available_scenarios():
        print(f"  - {name}")
    
    print("\n")
    
    # Generate and display a scenario
    scenario = generator.generate_scenario('patrol_ambush')
    generator.print_scenario(scenario)
    
    print("\n")
    
    # Generate random scenario
    random_scenario = generator.generate_random_scenario(duration=120, event_rate=0.1)
    generator.print_scenario(random_scenario)
