#!/usr/bin/env python3
"""
ARDS Detection and Obesity Analysis - Data Exploration
=====================================================

Research Question: How does obesity modify the relationship between early plateau pressures 
and clinical outcomes in ARDS patients, when ARDS onset is accurately detected using 
unstructured radiology reports?

Methodology: ARDSFlag + Berlin Definition implementation for MIMIC-IV
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# Data paths
DATA_BASE = '/Users/kavenchhikara/Desktop/CLIF/MIMIC-IV-3.1/physionet.org/files'
RAD_PATH = f'{DATA_BASE}/mimic-iv-note/2.2/note/radiology.csv.gz'
RAD_DETAIL_PATH = f'{DATA_BASE}/mimic-iv-note/2.2/note/radiology_detail.csv.gz'
MIMIC_BASE = f'{DATA_BASE}/mimiciv/3.1'

class ARDSDetector:
    """
    ARDS Detection using ARDSFlag methodology adapted for MIMIC-IV
    """
    
    def __init__(self):
        self.bilateral_patterns = [
            # Direct bilateral mentions
            r'\bbilateral\s+(?:opacit|infiltrat|consolidat)',
            r'(?:opacit|infiltrat|consolidat).{0,20}\bbilateral',
            
            # Anatomical bilateral patterns
            r'\bboth\s+(?:lung|lower\s+lobe|upper\s+lobe|base)',
            r'(?:right|left)\s+(?:and|&)\s+(?:left|right)\s+(?:lung|lobe|base)',
            
            # Diffuse/extensive patterns
            r'\bdiffuse\s+(?:opacit|infiltrat|consolidat)',
            r'\bextensive\s+(?:opacit|infiltrat|consolidat)',
            r'\bmultifocal\s+(?:opacit|infiltrat|consolidat)',
        ]
        
        # Exclusion patterns for negation
        self.exclusion_patterns = [
            r'no\s+(?:bilateral|diffuse|extensive)',
            r'without\s+(?:bilateral|diffuse|extensive)',
            r'absence\s+of\s+(?:bilateral|diffuse)',
            r'clear\s+(?:lung|bilateral)',
        ]
        
    def clean_text(self, text):
        """Clean and normalize radiology report text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize common medical terms
        text = re.sub(r'\bchf\b', 'congestive heart failure', text)
        text = re.sub(r'\bcopd\b', 'chronic obstructive pulmonary disease', text)
        text = re.sub(r'\bpe\b', 'pulmonary embolism', text)
        
        # Standardize anatomical terms
        text = re.sub(r'left\s+lung', 'leftlung', text)
        text = re.sub(r'right\s+lung', 'rightlung', text)
        text = re.sub(r'both\s+lungs', 'bilaterallungs', text)
        
        return text
    
    def detect_bilateral_opacities(self, text):
        """
        Detect bilateral opacities using rule-based pattern matching
        
        Returns: (has_bilateral_opacities: bool, confidence: float, matched_patterns: list)
        """
        clean_text = self.clean_text(text)
        
        # Check for exclusion patterns first
        for pattern in self.exclusion_patterns:
            if re.search(pattern, clean_text):
                return False, 0.0, ['excluded']
        
        # Check for bilateral opacity patterns
        matched_patterns = []
        for pattern in self.bilateral_patterns:
            if re.search(pattern, clean_text):
                matched_patterns.append(pattern)
        
        has_bilateral = len(matched_patterns) > 0
        confidence = min(len(matched_patterns) * 0.3, 1.0)  # Simple confidence scoring
        
        return has_bilateral, confidence, matched_patterns

def load_radiology_data(n_rows=None):
    """Load and process radiology reports"""
    print("Loading radiology data...")
    
    # Load main radiology table
    rad_df = pd.read_csv(RAD_PATH, nrows=n_rows)
    
    # Convert datetime columns
    rad_df['charttime'] = pd.to_datetime(rad_df['charttime'])
    rad_df['storetime'] = pd.to_datetime(rad_df['storetime'])
    
    # Filter for chest imaging
    chest_mask = rad_df['text'].str.contains(
        'chest.*x.*ray|chest.*pa.*lat|chest.*film|portable.*chest|thorax', 
        case=False, na=False
    )
    chest_reports = rad_df[chest_mask].copy()
    
    print(f"Total radiology reports: {len(rad_df):,}")
    print(f"Chest imaging reports: {len(chest_reports):,}")
    
    return chest_reports

def load_basic_patient_data():
    """Load basic patient demographics and admission data"""
    print("Loading patient demographics...")
    
    # Load patients table
    patients_df = pd.read_csv(f'{MIMIC_BASE}/hosp/patients.csv.gz')
    
    # Load admissions table  
    admissions_df = pd.read_csv(f'{MIMIC_BASE}/hosp/admissions.csv.gz')
    
    # Convert datetime columns
    patients_df['dod'] = pd.to_datetime(patients_df['dod'])
    admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
    admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])
    admissions_df['deathtime'] = pd.to_datetime(admissions_df['deathtime'])
    
    print(f"Patients: {len(patients_df):,}")
    print(f"Admissions: {len(admissions_df):,}")
    
    return patients_df, admissions_df

def main():
    """Main exploration workflow"""
    print("=" * 60)
    print("ARDS Detection and Obesity Analysis - Data Exploration")
    print("=" * 60)
    
    # Load data (start with subset for exploration)
    chest_reports = load_radiology_data(n_rows=10000)  # Start with 10k for speed
    patients_df, admissions_df = load_basic_patient_data()
    
    # Initialize ARDS detector
    detector = ARDSDetector()
    
    # Test bilateral opacity detection on sample
    print("\n" + "="*40)
    print("TESTING BILATERAL OPACITY DETECTION")
    print("="*40)
    
    sample_reports = chest_reports.head(10)
    for idx, row in sample_reports.iterrows():
        bilateral, confidence, patterns = detector.detect_bilateral_opacities(row['text'])
        print(f"\nReport ID: {row['note_id']}")
        print(f"Bilateral opacities: {bilateral} (confidence: {confidence:.2f})")
        if bilateral:
            print(f"Matched patterns: {len(patterns)}")
        print("Text preview:", row['text'][:200] + "...")
    
    # Basic statistics
    print("\n" + "="*40)
    print("BASIC DATA STATISTICS")
    print("="*40)
    
    print(f"Date range of chest reports: {chest_reports['charttime'].min()} to {chest_reports['charttime'].max()}")
    print(f"Unique patients with chest imaging: {chest_reports['subject_id'].nunique():,}")
    print(f"Unique admissions with chest imaging: {chest_reports['hadm_id'].nunique():,}")
    
    # Link with admission data
    chest_with_admit = chest_reports.merge(
        admissions_df[['hadm_id', 'subject_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag']], 
        on=['hadm_id', 'subject_id'], 
        how='inner'
    )
    
    print(f"Chest reports linked to admissions: {len(chest_with_admit):,}")
    print(f"Hospital mortality in linked cohort: {chest_with_admit['hospital_expire_flag'].mean():.1%}")
    
    print("\nData exploration completed successfully!")
    print("Next steps: ICU data linking and ventilator parameter extraction")

if __name__ == "__main__":
    main()