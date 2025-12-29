import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

class PredictiveMedicalAI:
    """
    AI-powered medical assessment:
    1. Predict injury severity from impact + vitals
    2. Automatic triage classification
    3. Resource allocation optimization
    4. Survival probability estimation
    """
    
    TRIAGE_CATEGORIES = {
        'IMMEDIATE': 'Red - Life-threatening, immediate intervention needed',
        'DELAYED': 'Yellow - Serious but stable, can wait 1-2 hours',
        'MINIMAL': 'Green - Minor injuries, walking wounded',
        'EXPECTANT': 'Black - Fatal injuries, palliative care only'
    }
    
    def __init__(self):
        # Load pre-trained injury prediction model
        self.injury_classifier = self._build_injury_model()
        self.survival_model = self._build_survival_model()
        
    def _build_injury_model(self):
        """Train injury classification model"""
        # In production: trained on military medical databases
        # Features: impact type, severity, vitals, time to treatment
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
        
        # Dummy training (would use real data)
        X_train = np.random.randn(1000, 10)
        y_train = np.random.randint(0, 4, 1000)
        model.fit(X_train, y_train)
        
        return model
    
    def _build_survival_model(self):
        """Survival probability estimator"""
        # Uses Cox proportional hazards or similar
        return None  # Placeholder
    
    async def assess_casualty(self, alert_data):
        """
        Comprehensive casualty assessment.
        
        Args:
            alert_data: Impact alert with vitals, location, impact type
            
        Returns:
            Medical assessment with triage category and treatment recommendations
        """
        # Extract features
        features = self._extract_medical_features(alert_data)
        
        # Predict injury severity
        injury_prediction = self._predict_injuries(features)
        
        # Triage classification
        triage_category = self._assign_triage(injury_prediction, alert_data)
        
        # Survival probability
        survival_prob = self._estimate_survival(features, injury_prediction)
        
        # Treatment recommendations
        treatments = self._recommend_treatments(injury_prediction, alert_data)
        
        # Golden hour countdown
        time_to_critical = self._calculate_golden_hour(injury_prediction)
        
        return {
            'triage_category': triage_category,
            'predicted_injuries': injury_prediction,
            'survival_probability': survival_prob,
            'recommended_treatments': treatments,
            'time_to_critical_minutes': time_to_critical,
            'evacuation_priority': self._calculate_evac_priority(triage_category, alert_data['location']),
            'required_resources': self._determine_resources(injury_prediction)
        }
    
    def _extract_medical_features(self, alert_data):
        """Extract features for medical ML model"""
        return np.array([
            self._encode_impact_type(alert_data['impact_type']),
            alert_data['severity'],
            alert_data['vitals']['heart_rate'],
            alert_data['vitals']['spo2'],
            alert_data['vitals']['breathing_rate'],
            alert_data.get('blood_pressure_systolic', 120),
            alert_data.get('blood_pressure_diastolic', 80),
            alert_data.get('time_since_impact', 0),
            alert_data.get('ambient_temperature', 25),
            alert_data.get('altitude', 0)
        ])
    
    def _encode_impact_type(self, impact_type):
        """Convert impact type to numeric feature"""
        encoding = {
            'blast': 1.0,
            'gunshot': 0.9,
            'artillery': 1.0,
            'vehicle_crash': 0.7,
            'fall': 0.5
        }
        return encoding.get(impact_type, 0.5)
    
    def _predict_injuries(self, features):
        """Predict specific injuries"""
        # Would use specialized models for each injury type
        
        # Simplified predictions
        injuries = {
            'traumatic_brain_injury': {
                'probability': 0.3 if features[0] > 0.8 else 0.1,
                'severity': 'moderate'
            },
            'internal_bleeding': {
                'probability': 0.4 if features[1] > 0.7 else 0.15,
                'severity': 'severe' if features[2] > 120 else 'moderate'
            },
            'fractures': {
                'probability': 0.6,
                'locations': ['right_femur', 'left_radius']
            },
            'blast_lung': {
                'probability': 0.25 if features[0] == 1.0 else 0.0,
                'severity': 'severe'
            },
            'hemorrhagic_shock': {
                'probability': 0.35 if features[2] > 130 or features[3] < 90 else 0.1,
                'stage': 2
            }
        }
        
        return injuries
    
    def _assign_triage(self, injuries, alert_data):
        """Assign triage category using START protocol"""
        vitals = alert_data['vitals']
        
        # Check respiratory rate
        if vitals['breathing_rate'] > 30 or vitals['breathing_rate'] < 10:
            return 'IMMEDIATE'
        
        # Check perfusion (SpO2 as proxy)
        if vitals['spo2'] < 90:
            return 'IMMEDIATE'
        
        # Check mental status (would need neurological data)
        # For now, infer from impact type and severity
        if alert_data['impact_type'] in ['blast', 'artillery'] and alert_data['severity'] > 0.8:
            return 'IMMEDIATE'
        
        # Check for life-threatening injuries
        if injuries['internal_bleeding']['probability'] > 0.5 or \
           injuries['blast_lung']['probability'] > 0.3:
            return 'IMMEDIATE'
        
        # Otherwise delayed or minimal
        if alert_data['severity'] > 0.5:
            return 'DELAYED'
        
        return 'MINIMAL'
    
    def _estimate_survival(self, features, injuries):
        """Estimate survival probability"""
        # Simplified model (would use actual survival prediction model)
        base_survival = 0.95
        
        # Adjust for injuries
        if injuries['traumatic_brain_injury']['probability'] > 0.5:
            base_survival -= 0.3
        
        if injuries['hemorrhagic_shock']['probability'] > 0.5:
            base_survival -= 0.4 * injuries['hemorrhagic_shock']['stage'] / 4
        
        if injuries['blast_lung']['probability'] > 0.3:
            base_survival -= 0.25
        
        # Adjust for vitals
        if features[2] > 140 or features[2] < 50:  # Heart rate
            base_survival -= 0.15
        
        if features[3] < 85:  # SpO2
            base_survival -= 0.2
        
        # Time factor (golden hour)
        time_since_impact = features[7] / 60  # Convert to hours
        if time_since_impact > 1:
            base_survival -= 0.1 * (time_since_impact - 1)
        
        return max(0.0, min(1.0, base_survival))
    
    def _recommend_treatments(self, injuries, alert_data):
        """Generate treatment protocol"""
        treatments = []
        
        if injuries['hemorrhagic_shock']['probability'] > 0.5:
            treatments.append({
                'intervention': 'Tourniquet application',
                'priority': 1,
                'time_critical': True,
                'instructions': 'Apply tourniquet 2-3 inches above wound, tighten until bleeding stops'
            })
            treatments.append({
                'intervention': 'IV fluid resuscitation',
                'priority': 2,
                'supplies': ['IV catheter', 'Lactated Ringers 1000ml']
            })
        
        if injuries['blast_lung']['probability'] > 0.3:
            treatments.append({
                'intervention': 'Oxygen therapy',
                'priority': 1,
                'target': 'SpO2 > 94%'
            })
            treatments.append({
                'intervention': 'Chest decompression',
                'priority': 1,
                'indication': 'Suspected pneumothorax'
            })
        
        if injuries['traumatic_brain_injury']['probability'] > 0.5:
            treatments.append({
                'intervention': 'Cervical spine stabilization',
                'priority': 1
            })
            treatments.append({
                'intervention': 'Avoid hypotension',
                'target': 'SBP > 110 mmHg'
            })
        
        # Pain management
        treatments.append({
            'intervention': 'Analgesia',
            'medication': 'Morphine 5-10mg IV' if alert_data['severity'] > 0.7 else 'Tramadol 50mg PO',
            'priority': 3
        })
        
        return treatments
    
    def _calculate_golden_hour(self, injuries):
        """Calculate time until critical (golden hour countdown)"""
        # Most critical injuries need treatment within 60 minutes
        base_time = 60  # minutes
        
        # Adjust based on injury severity
        if injuries['hemorrhagic_shock']['probability'] > 0.7:
            base_time = 20  # Severe bleeding: 20 minutes
        elif injuries['traumatic_brain_injury']['probability'] > 0.6:
            base_time = 30  # TBI: 30 minutes
        
        return base_time
    
    def _calculate_evac_priority(self, triage_category, location):
        """Prioritize evacuation order"""
        priority_scores = {
            'IMMEDIATE': 1,
            'DELAYED': 2,
            'MINIMAL': 3,
            'EXPECTANT': 4
        }
        
        base_priority = priority_scores[triage_category]
        
        # Adjust for location (if in hot zone, evacuate faster)
        # Would integrate with geo-fence and threat data
        
        return base_priority
    
    def _determine_resources(self, injuries):
        """Determine required medical resources"""
        resources = {
            'personnel': [],
            'equipment': [],
            'transport': None
        }
        
        # Personnel needed
        if injuries['traumatic_brain_injury']['probability'] > 0.5 or \
           injuries['internal_bleeding']['probability'] > 0.6:
            resources['personnel'].append('Trauma surgeon')
            resources['transport'] = 'Air ambulance (helicopter)'
        else:
            resources['personnel'].append('Paramedic')
            resources['transport'] = 'Ground ambulance'
        
        # Equipment
        if injuries['blast_lung']['probability'] > 0.3:
            resources['equipment'].extend(['Portable ventilator', 'Chest tube kit'])
        
        if injuries['hemorrhagic_shock']['probability'] > 0.5:
            resources['equipment'].extend(['Blood transfusion kit', 'Tourniquet', 'Hemostatic gauze'])
        
        return resources


class SquadMedicalCoordinator:
    """
    Coordinate medical response across entire squad.
    
    Features:
    - Optimal medic-to-casualty assignment
    - Resource allocation
    - Mass casualty incident management
    - Evacuation routing
    """
    
    def __init__(self):
        self.active_casualties = []
        self.available_medics = []
        self.medical_supplies = {}
        
    async def coordinate_response(self, casualties, medics, supplies):
        """Optimize medical response"""
        # Assignment problem: match medics to casualties
        assignments = self._optimal_assignment(casualties, medics)
        
        # Resource allocation
        supply_plan = self._allocate_supplies(casualties, supplies)
        
        # Evacuation sequence
        evac_order = self._plan_evacuation(casualties)
        
        return {
            'medic_assignments': assignments,
            'supply_allocation': supply_plan,
            'evacuation_sequence': evac_order
        }
    
    def _optimal_assignment(self, casualties, medics):
        """Hungarian algorithm for optimal medic-casualty matching"""
        # Simplified: assign nearest medic to highest priority casualty
        assignments = []
        
        sorted_casualties = sorted(
            casualties,
            key=lambda c: (c['triage_priority'], c['time_to_critical'])
        )
        
        for casualty in sorted_casualties:
            if not medics:
                break
            
            # Find nearest available medic
            nearest_medic = min(
                medics,
                key=lambda m: self._calculate_distance(m['position'], casualty['position'])
            )
            
            assignments.append({
                'medic_id': nearest_medic['id'],
                'casualty_id': casualty['device_id'],
                'eta_minutes': self._calculate_distance(nearest_medic['position'], casualty['position']) / 83.3  # Assume 5 km/h running speed
            })
            
            medics.remove(nearest_medic)
        
        return assignments
    
    def _allocate_supplies(self, casualties, supplies):
        """Distribute medical supplies based on predicted needs"""
        allocation = {}
        
        for casualty in casualties:
            needed = casualty['required_resources']['equipment']
            allocated = []
            
            for item in needed:
                if item in supplies and supplies[item] > 0:
                    allocated.append(item)
                    supplies[item] -= 1
            
            allocation[casualty['device_id']] = allocated
        
        return allocation
    
    def _plan_evacuation(self, casualties):
        """Determine evacuation order"""
        # Sort by triage priority and time criticality
        return sorted(
            casualties,
            key=lambda c: (c['triage_priority'], c['time_to_critical'], c['distance_to_base'])
        )
    
    def _calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions (meters)"""
        # Simplified 2D distance
        return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2])) * 111000  # degrees to meters
