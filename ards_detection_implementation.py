#!/usr/bin/env python3
"""
ARDS Detection Implementation using ARDSFlag Methodology
=======================================================

Implementation of Berlin Definition ARDS criteria for MIMIC-IV:
1. Bilateral opacities detection from radiology reports (NLP)
2. Hypoxemia: P/F ratio ≤ 300 
3. PEEP requirement: PEEP ≥ 5 cmH2O
4. Timing: Acute onset (within 1 week)
5. CHF exclusion
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# Data paths
DATA_BASE = '/Users/kavenchhikara/Desktop/CLIF/MIMIC-IV-3.1/physionet.org/files'
MIMIC_BASE = f'{DATA_BASE}/mimiciv/3.1'

class BerlinARDSDetector:
    """
    Berlin Definition ARDS Detection using ARDSFlag methodology
    """
    
    def __init__(self):
        # Enhanced bilateral opacity patterns from ARDSFlag
        self.bilateral_patterns = [
            # Direct bilateral opacity mentions
            r'bilateral\s+(?:ground[‐\s]?glass\s+)?(?:opacit|infiltrat|consolidat|shadowing)',
            r'(?:opacit|infiltrat|consolidat|shadowing).{0,30}bilateral',
            
            # Bilateral anatomical mentions
            r'bilateral\s+(?:lung|pulmonary|alveolar)',
            r'both\s+(?:lung|lower\s+lobe|upper\s+lobe|base)',
            r'(?:right|left)\s+(?:and|&|\+)\s+(?:left|right)\s+(?:lung|lobe|base)',
            
            # Diffuse/extensive patterns
            r'diffuse\s+(?:bilateral\s+)?(?:opacit|infiltrat|consolidat|ground[‐\s]?glass)',
            r'extensive\s+(?:bilateral\s+)?(?:opacit|infiltrat|consolidat)',
            r'multifocal\s+(?:opacit|infiltrat|consolidat)',
            r'widespread\s+(?:opacit|infiltrat|consolidat)',
            
            # Ground glass specific (common in ARDS)
            r'bilateral\s+ground[‐\s]?glass',
            r'diffuse\s+ground[‐\s]?glass',
            r'ground[‐\s]?glass\s+(?:opacit|change).{0,20}bilateral',
        ]
        
        # Exclusion patterns for negation
        self.exclusion_patterns = [
            r'no\s+(?:bilateral|diffuse|extensive|multifocal)',
            r'without\s+(?:bilateral|diffuse|extensive)',
            r'absence\s+of\s+(?:bilateral|diffuse)',
            r'clear\s+(?:lung|bilateral)',
            r'resolved\s+(?:bilateral|diffuse)',
            r'improving\s+(?:bilateral|diffuse)',
        ]
        
        # CHF/cardiogenic patterns for exclusion
        self.chf_patterns = [
            r'congestive\s+heart\s+failure',
            r'\bchf\b',
            r'cardiogenic\s+(?:edema|pulmonary)',
            r'heart\s+failure',
            r'cardiac\s+(?:failure|dysfunction)',
            r'left\s+(?:heart|ventricular)\s+failure',
            r'pulmonary\s+(?:edema|congestion).{0,30}cardiac',
        ]
    
    def clean_radiology_text(self, text):
        """Clean and normalize radiology report text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove patient identifiers and dates
        text = re.sub(r'___+', ' ', text)
        text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', ' ', text)
        
        # Normalize medical abbreviations
        text = re.sub(r'\bchf\b', 'congestive heart failure', text)
        text = re.sub(r'\bcopd\b', 'chronic obstructive pulmonary disease', text)
        text = re.sub(r'\bpe\b', 'pulmonary embolism', text)
        text = re.sub(r'\bGGO\b', 'ground glass opacity', text)
        
        # Standardize spacing around hyphens
        text = re.sub(r'ground-glass', 'ground glass', text)
        text = re.sub(r'ground‐glass', 'ground glass', text)
        
        return text
    
    def detect_bilateral_opacities(self, text):
        """
        Detect bilateral opacities using enhanced pattern matching
        
        Returns: dict with detection results
        """
        clean_text = self.clean_radiology_text(text)
        
        # Check for exclusion patterns first
        exclusions = []
        for pattern in self.exclusion_patterns:
            if re.search(pattern, clean_text):
                exclusions.append(pattern)
        
        if exclusions:
            return {
                'has_bilateral_opacities': False,
                'confidence': 0.0,
                'matched_patterns': [],
                'excluded_by': exclusions,
                'reason': 'negated'
            }
        
        # Check for bilateral opacity patterns
        matched_patterns = []
        for pattern in self.bilateral_patterns:
            matches = re.findall(pattern, clean_text)
            if matches:
                matched_patterns.append({
                    'pattern': pattern,
                    'matches': matches
                })
        
        has_bilateral = len(matched_patterns) > 0
        
        # Enhanced confidence scoring
        confidence = 0.0
        if has_bilateral:
            base_confidence = 0.3 * len(matched_patterns)
            
            # Bonus for specific high-confidence patterns
            high_conf_patterns = ['bilateral', 'diffuse', 'ground glass']
            for pattern_info in matched_patterns:
                for term in high_conf_patterns:
                    if term in pattern_info['pattern']:
                        confidence += 0.2
            
            confidence = min(confidence, 1.0)
        
        return {
            'has_bilateral_opacities': has_bilateral,
            'confidence': confidence,
            'matched_patterns': matched_patterns,
            'excluded_by': [],
            'reason': 'detected' if has_bilateral else 'not_found'
        }
    
    def detect_chf_exclusion(self, text):
        """
        Detect CHF/cardiogenic causes for exclusion
        
        Returns: dict with CHF detection results
        """
        clean_text = self.clean_radiology_text(text)
        
        matched_chf = []
        for pattern in self.chf_patterns:
            matches = re.findall(pattern, clean_text)
            if matches:
                matched_chf.append({
                    'pattern': pattern,
                    'matches': matches
                })
        
        has_chf = len(matched_chf) > 0
        
        return {
            'has_chf': has_chf,
            'matched_patterns': matched_chf,
            'reason': 'chf_detected' if has_chf else 'no_chf'
        }

def load_icu_chartevents(subject_ids=None, n_rows=None):
    """
    Load ICU chart events for ventilator parameters
    
    Key parameters we need:
    - Plateau Pressure
    - PEEP  
    - FiO2
    - PaO2 (for P/F ratio calculation)
    """
    print("Loading ICU chart events...")
    
    # Load items dictionary to find relevant item IDs
    items_df = pd.read_csv(f'{MIMIC_BASE}/icu/d_items.csv.gz')
    
    # Find ventilator-related items
    vent_items = items_df[
        items_df['label'].str.contains(
            'plateau|peep|fio2|pao2|respiratory|ventilator|vent', 
            case=False, na=False
        )
    ]
    
    print("Relevant ventilator items found:")
    for _, row in vent_items[['itemid', 'label', 'category']].iterrows():
        print(f"  {row['itemid']}: {row['label']} ({row['category']})")
    
    # Load chart events (start with subset)
    chartevents_df = pd.read_csv(f'{MIMIC_BASE}/icu/chartevents.csv.gz', nrows=n_rows)
    
    if subject_ids is not None:
        chartevents_df = chartevents_df[chartevents_df['subject_id'].isin(subject_ids)]
    
    # Filter for relevant items
    vent_chartevents = chartevents_df[
        chartevents_df['itemid'].isin(vent_items['itemid'])
    ].copy()
    
    # Convert charttime
    vent_chartevents['charttime'] = pd.to_datetime(vent_chartevents['charttime'])
    
    # Merge with item labels
    vent_chartevents = vent_chartevents.merge(
        items_df[['itemid', 'label', 'unitname']], 
        on='itemid', 
        how='left'
    )
    
    print(f"Ventilator chart events loaded: {len(vent_chartevents):,}")
    
    return vent_chartevents, vent_items

def analyze_bilateral_opacities(n_samples=1000):
    """
    Analyze bilateral opacity detection performance on sample data
    """
    print("=" * 60)
    print("BILATERAL OPACITY DETECTION ANALYSIS")
    print("=" * 60)
    
    # Load radiology data
    rad_path = f'{DATA_BASE}/mimic-iv-note/2.2/note/radiology.csv.gz'
    rad_df = pd.read_csv(rad_path, nrows=20000)  # Load larger sample
    
    # Filter for chest imaging
    chest_mask = rad_df['text'].str.contains(
        'chest.*x.*ray|chest.*pa.*lat|chest.*film|portable.*chest|thorax', 
        case=False, na=False
    )
    chest_reports = rad_df[chest_mask].copy()
    
    print(f"Analyzing {len(chest_reports)} chest reports...")
    
    # Initialize detector
    detector = BerlinARDSDetector()
    
    # Process reports
    results = []
    for idx, row in chest_reports.iterrows():
        bilateral_result = detector.detect_bilateral_opacities(row['text'])
        chf_result = detector.detect_chf_exclusion(row['text'])
        
        results.append({
            'note_id': row['note_id'],
            'subject_id': row['subject_id'],
            'hadm_id': row['hadm_id'],
            'charttime': row['charttime'],
            'has_bilateral_opacities': bilateral_result['has_bilateral_opacities'],
            'bilateral_confidence': bilateral_result['confidence'],
            'bilateral_patterns': len(bilateral_result['matched_patterns']),
            'has_chf': chf_result['has_chf'],
            'chf_patterns': len(chf_result['matched_patterns']),
            'text_length': len(row['text'])
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\nBILATERAL OPACITIES DETECTION RESULTS:")
    print(f"Total reports analyzed: {len(results_df):,}")
    print(f"Reports with bilateral opacities: {results_df['has_bilateral_opacities'].sum():,} ({results_df['has_bilateral_opacities'].mean():.1%})")
    print(f"Average confidence (positive cases): {results_df[results_df['has_bilateral_opacities']]['bilateral_confidence'].mean():.2f}")
    print(f"Reports with CHF mentions: {results_df['has_chf'].sum():,} ({results_df['has_chf'].mean():.1%})")
    
    # High confidence bilateral opacities
    high_conf = results_df[
        (results_df['has_bilateral_opacities']) & 
        (results_df['bilateral_confidence'] >= 0.5)
    ]
    print(f"High-confidence bilateral opacities (≥0.5): {len(high_conf):,}")
    
    # Show examples of detected cases
    print(f"\n=== EXAMPLES OF DETECTED BILATERAL OPACITIES ===")
    examples = chest_reports.merge(
        results_df[results_df['has_bilateral_opacities']].head(3),
        on='note_id'
    )
    
    for i, (_, row) in enumerate(examples.iterrows()):
        print(f"\n--- Example {i+1} (Confidence: {row['bilateral_confidence']:.2f}) ---")
        print(f"Report ID: {row['note_id']}")
        text_preview = row['text'][:500] + "..." if len(row['text']) > 500 else row['text']
        print(text_preview)
    
    return results_df

def main():
    """Main analysis workflow"""
    print("=" * 60)
    print("ARDS DETECTION IMPLEMENTATION - MIMIC-IV")
    print("=" * 60)
    
    # Step 1: Analyze bilateral opacity detection
    bilateral_results = analyze_bilateral_opacities()
    
    # Step 2: Identify patients with bilateral opacities for further analysis
    positive_subjects = bilateral_results[
        bilateral_results['has_bilateral_opacities']
    ]['subject_id'].unique()
    
    print(f"\nSubjects with bilateral opacities detected: {len(positive_subjects):,}")
    
    # Step 3: Load ventilator data for these subjects (sample)
    sample_subjects = positive_subjects[:50]  # Start with 50 subjects
    vent_data, vent_items = load_icu_chartevents(subject_ids=sample_subjects, n_rows=100000)
    
    print(f"\nAnalysis completed!")
    print(f"Next steps:")
    print(f"  1. Implement P/F ratio calculation")
    print(f"  2. Add PEEP and timing criteria")
    print(f"  3. Combine all Berlin Definition components")
    
    return bilateral_results, vent_data

if __name__ == "__main__":
    bilateral_results, vent_data = main()