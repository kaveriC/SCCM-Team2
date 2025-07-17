#!/usr/bin/env python3
"""
Configuration Management for ARDS Analysis Pipeline
===================================================

This module provides centralized configuration management for the ARDS analysis project.
It loads settings from config.yaml and provides easy access to configuration values.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Configuration manager for ARDS analysis pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to config.yaml file. If None, searches in project root.
        """
        self.project_root = Path(__file__).parent.parent
        
        if config_path is None:
            config_path = self.project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Set up derived paths
        self._setup_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please copy config.yaml.template to config.yaml and update the paths."
            )
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_paths(self):
        """Set up derived paths based on configuration"""
        # Data paths
        self.data_base = Path(self.config['data']['base_path'])
        self.mimic_note_path = Path(self.config['data']['mimic_note_path'])
        self.mimic_iv_path = Path(self.config['data']['mimic_iv_path'])
        
        # Specific file paths
        self.radiology_file = self.mimic_note_path / self.config['data']['radiology_file']
        self.chartevents_file = self.mimic_iv_path / self.config['data']['chartevents_file']
        self.icustays_file = self.mimic_iv_path / self.config['data']['icustays_file']
        self.d_items_file = self.mimic_iv_path / self.config['data']['d_items_file']
        self.patients_file = self.mimic_iv_path / self.config['data']['patients_file']
        self.admissions_file = self.mimic_iv_path / self.config['data']['admissions_file']
        self.labevents_file = self.mimic_iv_path / self.config['data']['labevents_file']
        
        # Output paths
        self.output_base = self.project_root / self.config['output']['base_dir']
        self.checkpoints_dir = self.output_base / self.config['output']['checkpoints_dir']
        self.figures_dir = self.output_base / self.config['output']['figures_dir']
        self.tables_dir = self.output_base / self.config['output']['tables_dir']
        self.analysis_dir = self.output_base / self.config['output']['analysis_dir']
        
        # Create output directories if they don't exist
        for dir_path in [self.output_base, self.checkpoints_dir, self.figures_dir, 
                        self.tables_dir, self.analysis_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key using dot notation (e.g., 'data.base_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_path(self, table_name: str) -> Path:
        """
        Get full path to a specific data file
        
        Args:
            table_name: Name of the table (e.g., 'radiology', 'chartevents')
            
        Returns:
            Full path to the data file
        """
        path_mapping = {
            'radiology': self.radiology_file,
            'chartevents': self.chartevents_file,
            'icustays': self.icustays_file,
            'd_items': self.d_items_file,
            'patients': self.patients_file,
            'admissions': self.admissions_file,
            'labevents': self.labevents_file
        }
        
        if table_name not in path_mapping:
            raise ValueError(f"Unknown table name: {table_name}")
        
        return path_mapping[table_name]
    
    def get_output_path(self, filename: str, subdir: str = None) -> Path:
        """
        Get full path to an output file
        
        Args:
            filename: Name of the output file
            subdir: Subdirectory (figures, tables, analysis, checkpoints)
            
        Returns:
            Full path to the output file
        """
        if subdir is None:
            return self.output_base / filename
        
        subdir_mapping = {
            'figures': self.figures_dir,
            'tables': self.tables_dir,
            'analysis': self.analysis_dir,
            'checkpoints': self.checkpoints_dir
        }
        
        if subdir not in subdir_mapping:
            raise ValueError(f"Unknown subdirectory: {subdir}")
        
        return subdir_mapping[subdir] / filename
    
    def validate_data_access(self) -> Dict[str, bool]:
        """
        Validate that all required data files are accessible
        
        Returns:
            Dictionary mapping file names to accessibility status
        """
        files_to_check = {
            'radiology': self.radiology_file,
            'chartevents': self.chartevents_file,
            'icustays': self.icustays_file,
            'd_items': self.d_items_file,
            'patients': self.patients_file,
            'admissions': self.admissions_file,
            'labevents': self.labevents_file
        }
        
        results = {}
        for name, path in files_to_check.items():
            results[name] = path.exists()
        
        return results
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        print("ARDS Analysis Configuration Summary")
        print("=" * 40)
        print(f"Project Root: {self.project_root}")
        print(f"Config File: {self.config_path}")
        print(f"Data Base Path: {self.data_base}")
        print(f"Output Base Path: {self.output_base}")
        print()
        
        print("Data File Accessibility:")
        validation = self.validate_data_access()
        for name, accessible in validation.items():
            status = "✓" if accessible else "✗"
            print(f"  {status} {name}: {accessible}")
        print()
        
        print("Processing Configuration:")
        print(f"  Max Processes: {self.get('processing.max_processes')}")
        print(f"  Test Sample Size: {self.get('processing.test_sample_size')}")
        print(f"  Batch Size: {self.get('processing.batch_size')}")
        print()
        
        print("Analysis Configuration:")
        print(f"  Obesity BMI Threshold: {self.get('analysis.bmi_thresholds.obesity')}")
        print(f"  ARDS P/F Ratio Threshold: {self.get('analysis.ards_criteria.pf_ratio_threshold')}")
        print(f"  PEEP Threshold: {self.get('analysis.ards_criteria.peep_threshold')}")

# Global configuration instance
_config = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration instance
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        ConfigManager instance
    """
    global _config
    if _config is None:
        _config = ConfigManager(config_path)
    return _config

def reload_config(config_path: Optional[str] = None):
    """
    Reload configuration from file
    
    Args:
        config_path: Path to config file
    """
    global _config
    _config = ConfigManager(config_path)

if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    config.print_config_summary()