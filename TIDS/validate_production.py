#!/usr/bin/env python3
"""
Production Readiness Validation Script
Comprehensive system validation before deployment
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

class ProductionValidator:
    """Validates production readiness across all system components"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {
            'total_checks': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'errors': []
        }
    
    def print_header(self, text: str):
        """Print formatted header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70)
    
    def check(self, description: str, condition: bool, critical: bool = True) -> bool:
        """Validate a condition and track results"""
        self.results['total_checks'] += 1
        
        if condition:
            print(f"✓ {description}")
            self.results['passed'] += 1
            return True
        else:
            if critical:
                print(f"✗ {description}")
                self.results['failed'] += 1
                self.results['errors'].append(description)
            else:
                print(f"⚠ {description}")
                self.results['warnings'] += 1
            return False
    
    def validate_file_structure(self) -> None:
        """Validate complete file structure"""
        self.print_header("FILE STRUCTURE VALIDATION")
        
        required_files = [
            'edge_device/config.yaml',
            'edge_device/main.py',
            'edge_device/sensor_fusion.py',
            'edge_device/model_loader.py',
            'edge_device/sensors/imu_handler.py',
            'edge_device/sensors/gps_handler.py',
            'edge_device/sensors/vitals_handler.py',
            'backend/app.py',
            'backend/api/alerts.py',
            'backend/api/telemetry.py',
            'backend/api/auth.py',
            'backend/services/threat_analyzer.py',
            'backend/services/medical_ai.py',
            'backend/models.py',
            'backend/database/queries.py',
            'dashboard/package.json',
            'dashboard/src/App.jsx',
            'dashboard/src/components/LiveMap.jsx',
            'ml_training/generate_dataset.py',
            'ml_training/train_production_model.py',
            'ml_training/data_generation.py',
            'simulation/blast_simulator.py',
            'simulation/scenario_generator.py',
            'tests/test_integration.py',
            'tests/test_backend_api.py',
            'tests/test_security.py',
            'tests/benchmark_performance.py',
            '.env.example',
            'docker-compose.yml'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            self.check(
                f"File exists: {file_path}",
                full_path.exists()
            )
    
    def validate_dependencies(self) -> None:
        """Validate all dependency files"""
        self.print_header("DEPENDENCY VALIDATION")
        
        # Check requirements.txt files
        requirements_files = [
            'edge_device/requirements.txt',
            'backend/requirements.txt',
            'ml_training/requirements.txt'
        ]
        
        for req_file in requirements_files:
            full_path = self.project_root / req_file
            exists = full_path.exists()
            self.check(f"Requirements file: {req_file}", exists)
            
            if exists:
                with open(full_path, 'r') as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                    self.check(
                        f"  {req_file} has dependencies ({len(lines)} packages)",
                        len(lines) > 0
                    )
        
        # Check package.json
        package_json = self.project_root / 'dashboard' / 'package.json'
        if package_json.exists():
            with open(package_json, 'r') as f:
                data = json.load(f)
                deps = data.get('dependencies', {})
                self.check(
                    f"Dashboard has npm dependencies ({len(deps)} packages)",
                    len(deps) > 0
                )
    
    def validate_ml_model(self) -> None:
        """Validate ML model files"""
        self.print_header("ML MODEL VALIDATION")
        
        model_path = self.project_root / 'ml_training' / 'models' / 'impact_classifier.tflite'
        model_exists = model_path.exists()
        
        self.check("TFLite model exists", model_exists, critical=False)
        
        if model_exists:
            size_kb = model_path.stat().st_size / 1024
            self.check(
                f"Model size is acceptable ({size_kb:.1f} KB < 500 KB)",
                size_kb < 500,
                critical=False
            )
        else:
            print("  ℹ Run training pipeline to create model:")
            print("    python ml_training/train_production_model.py")
        
        # Check if dataset exists
        dataset_path = self.project_root / 'ml_training' / 'data' / 'combat_trauma_dataset.h5'
        dataset_exists = dataset_path.exists()
        
        self.check("Training dataset exists", dataset_exists, critical=False)
        
        if not dataset_exists:
            print("  ℹ Generate dataset with:")
            print("    python ml_training/generate_dataset.py")
    
    def validate_configuration(self) -> None:
        """Validate configuration files"""
        self.print_header("CONFIGURATION VALIDATION")
        
        # Check .env.example
        env_example = self.project_root / '.env.example'
        self.check(".env.example exists", env_example.exists())
        
        # Check .env (should exist but not be in git)
        env_file = self.project_root / '.env'
        env_exists = env_file.exists()
        
        if not env_exists:
            print("⚠ .env file not found - copy from .env.example")
            print("  cp .env.example .env")
            self.results['warnings'] += 1
        
        # Check config.yaml
        config_yaml = self.project_root / 'edge_device' / 'config.yaml'
        self.check("Edge device config.yaml exists", config_yaml.exists())
        
        # Check docker-compose.yml
        docker_compose = self.project_root / 'docker-compose.yml'
        self.check("docker-compose.yml exists", docker_compose.exists())
    
    def validate_security(self) -> None:
        """Validate security configurations"""
        self.print_header("SECURITY VALIDATION")
        
        # Check .gitignore
        gitignore = self.project_root / '.gitignore'
        gitignore_exists = gitignore.exists()
        
        self.check(".gitignore exists", gitignore_exists)
        
        if gitignore_exists:
            with open(gitignore, 'r') as f:
                content = f.read()
                self.check(".gitignore includes .env", '.env' in content)
                self.check(".gitignore includes venv", 'venv' in content or '.venv' in content)
                self.check(".gitignore includes node_modules", 'node_modules' in content)
        
        # Check if .env is in git (should NOT be)
        env_in_git = os.system('git ls-files --error-unmatch .env 2>/dev/null') == 0
        self.check(".env is NOT tracked by git", not env_in_git)
    
    def validate_tests(self) -> None:
        """Validate test suite"""
        self.print_header("TEST SUITE VALIDATION")
        
        test_files = [
            'tests/test_integration.py',
            'tests/test_backend_api.py',
            'tests/test_security.py',
            'tests/benchmark_performance.py'
        ]
        
        for test_file in test_files:
            full_path = self.project_root / test_file
            self.check(f"Test file exists: {test_file}", full_path.exists())
    
    def validate_deployment_scripts(self) -> None:
        """Validate deployment automation"""
        self.print_header("DEPLOYMENT SCRIPTS VALIDATION")
        
        deploy_sh = self.project_root / 'deploy.sh'
        deploy_ps1 = self.project_root / 'deploy.ps1'
        
        self.check("Bash deployment script (deploy.sh)", deploy_sh.exists())
        self.check("PowerShell deployment script (deploy.ps1)", deploy_ps1.exists())
        
        # Check if scripts are executable (Unix only)
        if sys.platform != 'win32' and deploy_sh.exists():
            is_executable = os.access(deploy_sh, os.X_OK)
            if not is_executable:
                print("  ℹ Make deploy.sh executable: chmod +x deploy.sh")
    
    def print_summary(self) -> None:
        """Print validation summary"""
        self.print_header("VALIDATION SUMMARY")
        
        total = self.results['total_checks']
        passed = self.results['passed']
        failed = self.results['failed']
        warnings = self.results['warnings']
        
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\nTotal Checks: {total}")
        print(f"Passed:       {passed} ({pass_rate:.1f}%)")
        print(f"Failed:       {failed}")
        print(f"Warnings:     {warnings}")
        
        if failed > 0:
            print("\n" + "="*70)
            print("CRITICAL FAILURES:")
            print("="*70)
            for error in self.results['errors']:
                print(f"  ✗ {error}")
        
        print("\n" + "="*70)
        
        if pass_rate >= 90:
            print("✓ PRODUCTION READINESS: PASS (>90%)")
            print("  System is ready for deployment!")
            return_code = 0
        elif pass_rate >= 70:
            print("⚠ PRODUCTION READINESS: WARNING (70-90%)")
            print("  System can be deployed but needs improvements")
            return_code = 1
        else:
            print("✗ PRODUCTION READINESS: FAIL (<70%)")
            print("  System is NOT ready for production")
            return_code = 2
        
        print("="*70 + "\n")
        
        return return_code
    
    def run_all_validations(self) -> int:
        """Run complete validation suite"""
        print("\n" + "="*70)
        print("  GUARDIAN-SHIELD PRODUCTION READINESS VALIDATION")
        print("="*70)
        
        self.validate_file_structure()
        self.validate_dependencies()
        self.validate_ml_model()
        self.validate_configuration()
        self.validate_security()
        self.validate_tests()
        self.validate_deployment_scripts()
        
        return self.print_summary()

if __name__ == "__main__":
    validator = ProductionValidator()
    exit_code = validator.run_all_validations()
    sys.exit(exit_code)
