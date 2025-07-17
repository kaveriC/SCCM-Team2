#!/usr/bin/env python3
"""
ARDS Detection Pipeline - Production Script
===========================================

This script runs the complete ARDS detection pipeline with:
- Multiprocessing for performance
- Checkpointing for reliability
- Comprehensive logging and monitoring
- Summary figures and tables
- STROBE diagram generation

Usage:
    python run_ards_detection.py --test     # Quick test run
    python run_ards_detection.py --full     # Full dataset overnight run
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import psutil
import gc
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import configuration
from src.config import get_config

# Load configuration
config = get_config()

# Set up paths from config
RAD_PATH = config.get_data_path('radiology')
OUTPUT_DIR = config.output_base
CHECKPOINT_DIR = config.checkpoints_dir
FIGURES_DIR = config.figures_dir
TABLES_DIR = config.tables_dir

class BerlinARDSDetector:
    """Enhanced ARDS Detector with Berlin Definition criteria"""
    
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
        """Detect bilateral opacities using enhanced pattern matching"""
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
        """Detect CHF/cardiogenic causes for exclusion"""
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

def setup_logging(test_mode=False):
    """Setup comprehensive logging"""
    log_level = logging.DEBUG if test_mode else logging.INFO
    log_file = OUTPUT_DIR / f"ards_detection_{'test' if test_mode else 'full'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting ARDS Detection Pipeline - {'Test' if test_mode else 'Full'} Mode")
    logger.info(f"Log file: {log_file}")
    
    return logger

def check_memory_usage():
    """Monitor memory usage"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / 1e9,
        'used_gb': memory.used / 1e9,
        'available_gb': memory.available / 1e9,
        'percent': memory.percent
    }

def process_report_batch(batch_data):
    """Process a batch of reports for bilateral opacity detection"""
    detector = BerlinARDSDetector()
    results = []
    
    for _, row in batch_data.iterrows():
        try:
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
                'text_length': len(row['text']),
                'exclusion_reason': bilateral_result['reason']
            })
        except Exception as e:
            logging.warning(f"Error processing report {row['note_id']}: {e}")
            # Add placeholder result for failed processing
            results.append({
                'note_id': row['note_id'],
                'subject_id': row['subject_id'],
                'hadm_id': row['hadm_id'],
                'charttime': row['charttime'],
                'has_bilateral_opacities': False,
                'bilateral_confidence': 0.0,
                'bilateral_patterns': 0,
                'has_chf': False,
                'chf_patterns': 0,
                'text_length': len(row['text']) if pd.notna(row['text']) else 0,
                'exclusion_reason': 'processing_error'
            })
    
    return results

def load_and_filter_radiology_data(test_mode=False, logger=None):
    """Load and filter radiology data for chest imaging"""
    logger.info("Loading radiology data...")
    
    # Load data with size limit for testing
    test_sample_size = config.get('processing.test_sample_size', 5000)
    if test_mode:
        rad_df = pd.read_csv(RAD_PATH, nrows=test_sample_size)
        logger.info(f"Test mode: Loading {test_sample_size:,} radiology reports")
    else:
        rad_df = pd.read_csv(RAD_PATH)
        logger.info(f"Full mode: Loading {len(rad_df):,} radiology reports")
    
    # Convert datetime columns
    rad_df['charttime'] = pd.to_datetime(rad_df['charttime'])
    rad_df['storetime'] = pd.to_datetime(rad_df['storetime'])
    
    # Filter for chest imaging
    chest_mask = rad_df['text'].str.contains(
        'chest.*x.*ray|chest.*pa.*lat|chest.*film|portable.*chest|thorax', 
        case=False, na=False
    )
    chest_reports = rad_df[chest_mask].copy()
    
    logger.info(f"Chest imaging reports identified: {len(chest_reports):,}")
    logger.info(f"Chest imaging percentage: {len(chest_reports)/len(rad_df):.1%}")
    
    # Save checkpoint
    checkpoint_file = CHECKPOINT_DIR / f"chest_reports_{'test' if test_mode else 'full'}.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(chest_reports, f)
    logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    return chest_reports

def process_reports_multiprocessing(chest_reports, test_mode=False, logger=None):
    """Process reports using multiprocessing"""
    logger.info("Starting bilateral opacity detection with multiprocessing...")
    
    # Determine number of processes from config
    max_processes = config.get('processing.max_processes', 32) if not test_mode else config.get('processing.test_max_processes', 4)
    n_processes = min(cpu_count(), max_processes, len(chest_reports) // 100)
    n_processes = max(1, n_processes)  # At least 1 process
    
    logger.info(f"Using {n_processes} processes for {len(chest_reports):,} reports")
    
    # Split data into batches
    batch_size = max(1, len(chest_reports) // n_processes)
    batches = [chest_reports.iloc[i:i + batch_size] for i in range(0, len(chest_reports), batch_size)]
    
    logger.info(f"Processing in {len(batches)} batches of ~{batch_size} reports each")
    
    # Process batches
    all_results = []
    
    if n_processes == 1:
        # Single process for small datasets
        for i, batch in enumerate(tqdm(batches, desc="Processing batches")):
            logger.info(f"Processing batch {i+1}/{len(batches)}")
            batch_results = process_report_batch(batch)
            all_results.extend(batch_results)
            
            # Memory check
            mem_info = check_memory_usage()
            logger.info(f"Memory usage: {mem_info['percent']:.1f}% ({mem_info['used_gb']:.1f}GB)")
            
            # Save intermediate checkpoint every N batches
            checkpoint_freq = config.get('processing.checkpoint_frequency', 5)
            if (i + 1) % checkpoint_freq == 0:
                checkpoint_file = CHECKPOINT_DIR / f"intermediate_results_batch_{i+1}.pkl"
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(all_results, f)
                logger.info(f"Intermediate checkpoint saved: {checkpoint_file}")
    else:
        # Multiprocessing
        with Pool(processes=n_processes) as pool:
            batch_results = list(tqdm(
                pool.imap(process_report_batch, batches),
                total=len(batches),
                desc="Processing batches"
            ))
            
            # Flatten results
            for batch_result in batch_results:
                all_results.extend(batch_result)
    
    results_df = pd.DataFrame(all_results)
    
    # Final checkpoint
    checkpoint_file = CHECKPOINT_DIR / f"ards_detection_results_{'test' if test_mode else 'full'}.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(results_df, f)
    logger.info(f"Final results checkpoint saved: {checkpoint_file}")
    
    return results_df

def generate_summary_statistics(results_df, chest_reports, test_mode=False, logger=None):
    """Generate comprehensive summary statistics and tables"""
    logger.info("Generating summary statistics...")
    
    # Basic statistics
    stats = {
        'total_reports_analyzed': len(results_df),
        'chest_reports_total': len(chest_reports),
        'bilateral_opacities_detected': results_df['has_bilateral_opacities'].sum(),
        'bilateral_opacities_rate': results_df['has_bilateral_opacities'].mean(),
        'chf_mentions': results_df['has_chf'].sum(),
        'chf_rate': results_df['has_chf'].mean(),
        'unique_subjects': results_df['subject_id'].nunique(),
        'unique_admissions': results_df['hadm_id'].nunique(),
    }
    
    # Confidence distribution for positive cases
    positive_cases = results_df[results_df['has_bilateral_opacities']]
    if len(positive_cases) > 0:
        stats.update({
            'positive_cases_count': len(positive_cases),
            'mean_confidence': positive_cases['bilateral_confidence'].mean(),
            'median_confidence': positive_cases['bilateral_confidence'].median(),
            'high_confidence_cases': (positive_cases['bilateral_confidence'] >= 0.5).sum(),
            'unique_subjects_bilateral': positive_cases['subject_id'].nunique(),
            'unique_admissions_bilateral': positive_cases['hadm_id'].nunique(),
        })
    
    # Save summary statistics
    stats_file = TABLES_DIR / f"summary_statistics_{'test' if test_mode else 'full'}.json"
    import json
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.info(f"Summary statistics saved: {stats_file}")
    
    # Create Table 1: Cohort characteristics
    table1_data = {
        'Characteristic': [
            'Total radiology reports analyzed',
            'Chest imaging reports identified',
            'Bilateral opacities detected',
            'Reports with CHF mentions',
            'High-confidence bilateral opacities (≥0.5)',
            'Unique subjects with bilateral opacities',
            'Unique admissions with bilateral opacities'
        ],
        'N (%)': [
            f"{stats['total_reports_analyzed']:,}",
            f"{stats['chest_reports_total']:,} ({stats['chest_reports_total']/stats['total_reports_analyzed']*100:.1f}%)",
            f"{stats['bilateral_opacities_detected']:,} ({stats['bilateral_opacities_rate']*100:.1f}%)",
            f"{stats['chf_mentions']:,} ({stats['chf_rate']*100:.1f}%)",
            f"{stats.get('high_confidence_cases', 0):,}" + (f" ({stats['high_confidence_cases']/stats['positive_cases_count']*100:.1f}%)" if stats.get('positive_cases_count', 0) > 0 else ""),
            f"{stats.get('unique_subjects_bilateral', 0):,}",
            f"{stats.get('unique_admissions_bilateral', 0):,}"
        ]
    }
    
    table1_df = pd.DataFrame(table1_data)
    table1_file = TABLES_DIR / f"table1_cohort_characteristics_{'test' if test_mode else 'full'}.csv"
    table1_df.to_csv(table1_file, index=False)
    
    logger.info(f"Table 1 (Cohort Characteristics) saved: {table1_file}")
    
    return stats

def create_strobe_diagram(stats, test_mode=False, logger=None):
    """Create STROBE diagram for cohort identification"""
    logger.info("Creating STROBE diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # STROBE diagram boxes and text
    boxes = [
        # Initial dataset
        {'xy': (5, 18), 'width': 4, 'height': 1, 
         'text': f"Total MIMIC-IV radiology reports\n(n = {stats['total_reports_analyzed']:,})", 'color': 'lightblue'},
        
        # Chest imaging filter
        {'xy': (5, 16), 'width': 4, 'height': 1,
         'text': f"Chest imaging reports identified\n(n = {stats['chest_reports_total']:,})", 'color': 'lightgreen'},
        
        # Excluded box
        {'xy': (0.5, 16), 'width': 3, 'height': 1,
         'text': f"Excluded: Non-chest imaging\n(n = {stats['total_reports_analyzed'] - stats['chest_reports_total']:,})", 'color': 'lightcoral'},
        
        # ARDS detection
        {'xy': (5, 14), 'width': 4, 'height': 1,
         'text': f"Bilateral opacities detected\n(n = {stats['bilateral_opacities_detected']:,})", 'color': 'lightyellow'},
        
        # Excluded - no bilateral opacities
        {'xy': (0.5, 14), 'width': 3, 'height': 1,
         'text': f"Excluded: No bilateral opacities\n(n = {stats['chest_reports_total'] - stats['bilateral_opacities_detected']:,})", 'color': 'lightcoral'},
        
        # High confidence cases
        {'xy': (5, 12), 'width': 4, 'height': 1,
         'text': f"High-confidence bilateral opacities\n(n = {stats.get('high_confidence_cases', 0):,})", 'color': 'gold'},
        
        # CHF exclusions
        {'xy': (0.5, 12), 'width': 3, 'height': 1,
         'text': f"Reports with CHF mentions\n(n = {stats['chf_mentions']:,})", 'color': 'lightsalmon'},
        
        # Final cohort
        {'xy': (5, 10), 'width': 4, 'height': 1.5,
         'text': f"Final ARDS candidate cohort\n\nUnique subjects: {stats.get('unique_subjects_bilateral', 0):,}\nUnique admissions: {stats.get('unique_admissions_bilateral', 0):,}", 'color': 'lightsteelblue'},
    ]
    
    # Draw boxes
    for box in boxes:
        rect = plt.Rectangle((box['xy'][0] - box['width']/2, box['xy'][1] - box['height']/2), 
                           box['width'], box['height'], 
                           facecolor=box['color'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(box['xy'][0], box['xy'][1], box['text'], 
                ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw arrows
    arrows = [
        ((5, 17.5), (5, 16.5)),  # Total to chest imaging
        ((5, 15.5), (5, 14.5)),  # Chest imaging to bilateral
        ((5, 13.5), (5, 12.5)),  # Bilateral to high confidence
        ((5, 11.25), (5, 10.75)),  # High confidence to final
        
        # Exclusion arrows
        ((3, 17), (2, 16.5)),    # To non-chest exclusion
        ((3, 15), (2, 14.5)),    # To no bilateral exclusion
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Title
    ax.text(5, 19.5, 'STROBE Diagram: ARDS Cohort Identification', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Date and mode
    mode_text = "Test Mode" if test_mode else "Full Dataset"
    ax.text(5, 8.5, f'{mode_text} - Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
            ha='center', va='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save figure
    strobe_file = FIGURES_DIR / f"strobe_diagram_{'test' if test_mode else 'full'}.png"
    plt.savefig(strobe_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"STROBE diagram saved: {strobe_file}")

def create_summary_figures(results_df, test_mode=False, logger=None):
    """Create summary figures for the analysis"""
    logger.info("Creating summary figures...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bilateral opacity detection rates
    detection_counts = results_df['has_bilateral_opacities'].value_counts()
    axes[0, 0].pie(detection_counts.values, labels=['No Bilateral Opacities', 'Bilateral Opacities Detected'], 
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Bilateral Opacity Detection Results')
    
    # 2. Confidence distribution for positive cases
    positive_cases = results_df[results_df['has_bilateral_opacities']]
    if len(positive_cases) > 0:
        axes[0, 1].hist(positive_cases['bilateral_confidence'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(positive_cases['bilateral_confidence'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {positive_cases["bilateral_confidence"].mean():.2f}')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Number of Reports')
        axes[0, 1].set_title('Confidence Distribution (Positive Cases)')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No positive cases found', ha='center', va='center')
        axes[0, 1].set_title('Confidence Distribution (No Data)')
    
    # 3. CHF vs Bilateral Opacities
    chf_bilateral_crosstab = pd.crosstab(results_df['has_chf'], results_df['has_bilateral_opacities'])
    sns.heatmap(chf_bilateral_crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('CHF Mentions vs Bilateral Opacities')
    axes[1, 0].set_xlabel('Has Bilateral Opacities')
    axes[1, 0].set_ylabel('Has CHF Mention')
    
    # 4. Report length distribution
    axes[1, 1].hist(results_df['text_length'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Report Length (characters)')
    axes[1, 1].set_ylabel('Number of Reports')
    axes[1, 1].set_title('Radiology Report Length Distribution')
    
    plt.tight_layout()
    
    # Save figure
    summary_file = FIGURES_DIR / f"summary_figures_{'test' if test_mode else 'full'}.png"
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Summary figures saved: {summary_file}")

def save_final_outputs(results_df, test_mode=False, logger=None):
    """Save final analysis outputs"""
    logger.info("Saving final outputs...")
    
    # Save main results
    results_file = OUTPUT_DIR / f"bilateral_opacity_detection_results_{'test' if test_mode else 'full'}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Main results saved: {results_file}")
    
    # Save positive cases only
    positive_cases = results_df[results_df['has_bilateral_opacities']]
    if len(positive_cases) > 0:
        positive_file = OUTPUT_DIR / f"ards_candidates_{'test' if test_mode else 'full'}.csv"
        positive_cases.to_csv(positive_file, index=False)
        logger.info(f"ARDS candidates saved: {positive_file}")
    
    # Save subject list for next pipeline stage
    subject_list = positive_cases['subject_id'].unique()
    subject_file = OUTPUT_DIR / f"ards_subject_list_{'test' if test_mode else 'full'}.pkl"
    with open(subject_file, 'wb') as f:
        pickle.dump(subject_list, f)
    logger.info(f"Subject list for next stage saved: {subject_file}")

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ARDS Detection Pipeline')
    parser.add_argument('--test', action='store_true', help='Run in test mode (5k reports)')
    parser.add_argument('--full', action='store_true', help='Run on full dataset')
    
    args = parser.parse_args()
    
    if not args.test and not args.full:
        print("Please specify either --test or --full mode")
        sys.exit(1)
    
    test_mode = args.test
    
    # Setup logging
    logger = setup_logging(test_mode)
    
    # Log system information
    mem_info = check_memory_usage()
    logger.info(f"System memory: {mem_info['total_gb']:.1f}GB total, {mem_info['available_gb']:.1f}GB available")
    logger.info(f"CPU cores available: {cpu_count()}")
    
    start_time = datetime.now()
    logger.info(f"Pipeline started at: {start_time}")
    
    try:
        # Step 1: Load and filter radiology data
        chest_reports = load_and_filter_radiology_data(test_mode, logger)
        
        # Step 2: Process reports with multiprocessing
        results_df = process_reports_multiprocessing(chest_reports, test_mode, logger)
        
        # Step 3: Generate summary statistics
        stats = generate_summary_statistics(results_df, chest_reports, test_mode, logger)
        
        # Step 4: Create STROBE diagram
        create_strobe_diagram(stats, test_mode, logger)
        
        # Step 5: Create summary figures
        create_summary_figures(results_df, test_mode, logger)
        
        # Step 6: Save final outputs
        save_final_outputs(results_df, test_mode, logger)
        
        # Final statistics
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Duration: {duration}")
        logger.info(f"Total reports processed: {len(results_df):,}")
        logger.info(f"ARDS candidates identified: {results_df['has_bilateral_opacities'].sum():,}")
        logger.info(f"Processing rate: {len(results_df)/duration.total_seconds():.1f} reports/second")
        
        final_mem = check_memory_usage()
        logger.info(f"Final memory usage: {final_mem['percent']:.1f}% ({final_mem['used_gb']:.1f}GB)")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()