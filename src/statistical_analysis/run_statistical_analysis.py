#!/usr/bin/env python3
"""
Statistical Analysis Pipeline - Production Script
===============================================

This script performs comprehensive statistical analysis for the ARDS obesity study:
- Descriptive statistics and cohort characteristics
- Obesity-plateau pressure interaction analysis
- Multivariable regression models
- Survival analysis
- Sensitivity analyses

Usage:
    python run_statistical_analysis.py --test     # Quick test run
    python run_statistical_analysis.py --full     # Full analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import argparse
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical packages
import scipy.stats as stats
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from statsmodels.stats.contingency_tables import mcnemar
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import configuration
from src.config import get_config

# Load configuration
config = get_config()

# Set up paths from config
OUTPUT_DIR = config.output_base
FIGURES_DIR = config.figures_dir
TABLES_DIR = config.tables_dir
ANALYSIS_DIR = config.analysis_dir

def setup_logging(test_mode=False):
    """Setup comprehensive logging"""
    log_level = logging.INFO
    log_file = OUTPUT_DIR / f"statistical_analysis_{'test' if test_mode else 'full'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Statistical Analysis Pipeline - {'Test' if test_mode else 'Full'} Mode")
    logger.info(f"Log file: {log_file}")
    
    return logger

def load_analysis_data(test_mode=False, logger=None):
    """Load the analysis-ready dataset from previous pipeline stages"""
    logger.info("Loading analysis-ready dataset...")
    
    # Load main analysis dataset
    analysis_file = OUTPUT_DIR / f"analysis_ready_dataset_{'test' if test_mode else 'full'}.csv"
    
    if not analysis_file.exists():
        logger.error(f"Analysis dataset not found: {analysis_file}")
        logger.info("Please run the previous pipeline stages first:")
        logger.info("1. python src/ards_detection/run_ards_detection.py --full")
        logger.info("2. python src/data_extraction/run_ventilator_extraction.py --full")
        return None
    
    df = pd.read_csv(analysis_file)
    logger.info(f"Loaded analysis dataset: {df.shape}")
    
    # Load additional datasets for comprehensive analysis
    additional_files = {
        'berlin_criteria': OUTPUT_DIR / f"berlin_criteria_assessment_{'test' if test_mode else 'full'}.csv",
        'clinical_outcomes': OUTPUT_DIR / f"clinical_outcomes_{'test' if test_mode else 'full'}.csv",
        'bmi_classification': OUTPUT_DIR / f"bmi_classification_{'test' if test_mode else 'full'}.csv"
    }
    
    additional_data = {}
    for name, file_path in additional_files.items():
        if file_path.exists():
            additional_data[name] = pd.read_csv(file_path)
            logger.info(f"Loaded {name}: {additional_data[name].shape}")
        else:
            logger.warning(f"Optional file not found: {file_path}")
    
    return df, additional_data

def generate_descriptive_statistics(df, additional_data, test_mode=False, logger=None):
    """Generate comprehensive descriptive statistics"""
    logger.info("Generating descriptive statistics...")
    
    # Basic cohort characteristics
    total_subjects = len(df)
    logger.info(f"Total subjects in analysis: {total_subjects}")
    
    if total_subjects == 0:
        logger.error("No subjects available for analysis")
        return None
    
    # Create comprehensive Table 1
    table1_data = []
    
    # Demographics
    if 'age' in df.columns:
        age_mean = df['age'].mean()
        age_std = df['age'].std()
        table1_data.append(['Age (years)', f"{age_mean:.1f} ± {age_std:.1f}"])
    
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        if 'M' in gender_counts:
            male_pct = gender_counts['M'] / len(df) * 100
            table1_data.append(['Male gender', f"{gender_counts['M']} ({male_pct:.1f}%)"])
    
    # BMI and obesity
    if 'bmi' in df.columns:
        bmi_mean = df['bmi'].mean()
        bmi_std = df['bmi'].std()
        table1_data.append(['BMI (kg/m²)', f"{bmi_mean:.1f} ± {bmi_std:.1f}"])
    
    if 'obese' in df.columns:
        obese_count = df['obese'].sum()
        obese_pct = obese_count / len(df) * 100
        table1_data.append(['Obese (BMI ≥30)', f"{obese_count} ({obese_pct:.1f}%)"])
    
    # Ventilator parameters
    if 'mean_plateau_pressure' in df.columns:
        plateau_data = df['mean_plateau_pressure'].dropna()
        if len(plateau_data) > 0:
            plateau_mean = plateau_data.mean()
            plateau_std = plateau_data.std()
            table1_data.append(['Plateau pressure (cmH2O)', f"{plateau_mean:.1f} ± {plateau_std:.1f}"])
    
    if 'mean_peep' in df.columns:
        peep_data = df['mean_peep'].dropna()
        if len(peep_data) > 0:
            peep_mean = peep_data.mean()
            peep_std = peep_data.std()
            table1_data.append(['PEEP (cmH2O)', f"{peep_mean:.1f} ± {peep_std:.1f}"])
    
    # Clinical outcomes
    if 'icu_mortality' in df.columns:
        icu_mort_count = df['icu_mortality'].sum()
        icu_mort_pct = icu_mort_count / len(df) * 100
        table1_data.append(['ICU mortality', f"{icu_mort_count} ({icu_mort_pct:.1f}%)"])
    
    if 'hospital_mortality' in df.columns:
        hosp_mort_count = df['hospital_mortality'].sum()
        hosp_mort_pct = hosp_mort_count / len(df) * 100
        table1_data.append(['Hospital mortality', f"{hosp_mort_count} ({hosp_mort_pct:.1f}%)"])
    
    if 'ventilator_free_days_28' in df.columns:
        vfd_data = df['ventilator_free_days_28'].dropna()
        if len(vfd_data) > 0:
            vfd_mean = vfd_data.mean()
            vfd_std = vfd_data.std()
            table1_data.append(['28-day ventilator-free days', f"{vfd_mean:.1f} ± {vfd_std:.1f}"])
    
    # Save Table 1
    if table1_data:
        table1_df = pd.DataFrame(table1_data, columns=['Characteristic', 'Value'])
        table1_file = TABLES_DIR / f"table1_descriptive_statistics_{'test' if test_mode else 'full'}.csv"
        table1_df.to_csv(table1_file, index=False)
        logger.info(f"Table 1 saved: {table1_file}")
    
    return table1_df

def analyze_obesity_plateau_interaction(df, test_mode=False, logger=None):
    """Analyze obesity-plateau pressure interaction on outcomes"""
    logger.info("Analyzing obesity-plateau pressure interaction...")
    
    # Check if we have sufficient data
    if len(df) < 10:
        logger.warning("Insufficient data for interaction analysis")
        return None
    
    # Required columns
    required_cols = ['obese', 'mean_plateau_pressure', 'icu_mortality']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns for interaction analysis: {missing_cols}")
        return None
    
    # Filter for complete data
    analysis_df = df[required_cols].dropna()
    logger.info(f"Subjects with complete data for interaction analysis: {len(analysis_df)}")
    
    if len(analysis_df) < 10:
        logger.warning("Insufficient complete data for interaction analysis")
        return None
    
    # Interaction analysis results
    results = {}
    
    # 1. Stratified analysis by obesity status
    logger.info("Performing stratified analysis by obesity status...")
    
    obese_df = analysis_df[analysis_df['obese'] == True]
    non_obese_df = analysis_df[analysis_df['obese'] == False]
    
    results['obese_n'] = len(obese_df)
    results['non_obese_n'] = len(non_obese_df)
    
    logger.info(f"Obese subjects: {results['obese_n']}")
    logger.info(f"Non-obese subjects: {results['non_obese_n']}")
    
    # 2. Correlation analysis
    if len(obese_df) > 3:
        obese_corr = obese_df['mean_plateau_pressure'].corr(obese_df['icu_mortality'])
        results['obese_correlation'] = obese_corr
        logger.info(f"Plateau pressure-mortality correlation in obese: {obese_corr:.3f}")
    
    if len(non_obese_df) > 3:
        non_obese_corr = non_obese_df['mean_plateau_pressure'].corr(non_obese_df['icu_mortality'])
        results['non_obese_correlation'] = non_obese_corr
        logger.info(f"Plateau pressure-mortality correlation in non-obese: {non_obese_corr:.3f}")
    
    # 3. Logistic regression with interaction term
    try:
        logger.info("Fitting logistic regression model with interaction...")
        
        # Standardize plateau pressure
        analysis_df['plateau_std'] = (analysis_df['mean_plateau_pressure'] - analysis_df['mean_plateau_pressure'].mean()) / analysis_df['mean_plateau_pressure'].std()
        analysis_df['obesity_int'] = analysis_df['obese'].astype(int)
        analysis_df['interaction_term'] = analysis_df['obesity_int'] * analysis_df['plateau_std']
        
        # Fit model
        X = analysis_df[['obesity_int', 'plateau_std', 'interaction_term']]
        y = analysis_df['icu_mortality']
        
        # Add constant
        X = sm.add_constant(X)
        
        model = sm.Logit(y, X).fit(disp=0)
        results['model_summary'] = model.summary()
        
        # Extract key results
        results['obesity_coef'] = model.params['obesity_int']
        results['plateau_coef'] = model.params['plateau_std']
        results['interaction_coef'] = model.params['interaction_term']
        
        results['obesity_pvalue'] = model.pvalues['obesity_int']
        results['plateau_pvalue'] = model.pvalues['plateau_std']
        results['interaction_pvalue'] = model.pvalues['interaction_term']
        
        logger.info(f"Obesity effect: β={results['obesity_coef']:.3f}, p={results['obesity_pvalue']:.3f}")
        logger.info(f"Plateau pressure effect: β={results['plateau_coef']:.3f}, p={results['plateau_pvalue']:.3f}")
        logger.info(f"Interaction effect: β={results['interaction_coef']:.3f}, p={results['interaction_pvalue']:.3f}")
        
        # Save model results
        model_file = ANALYSIS_DIR / f"interaction_model_{'test' if test_mode else 'full'}.txt"
        with open(model_file, 'w') as f:
            f.write(str(model.summary()))
        logger.info(f"Model results saved: {model_file}")
        
        # Save detailed results to CSV and JSON
        detailed_results = save_statistical_results(results, model, test_mode, logger)
        
        # Store detailed results in the main results dict
        results['detailed_results'] = detailed_results
        results['model_summary'] = detailed_results['model_summary']
        
    except Exception as e:
        logger.error(f"Error in logistic regression: {e}")
        results['model_error'] = str(e)
    
    return results

def save_statistical_results(results, model, test_mode=False, logger=None):
    """Save statistical analysis results to CSV and JSON files"""
    logger.info("Saving statistical results to structured files...")
    
    # Create detailed results dictionary
    detailed_results = {
        'model_summary': {
            'n_observations': int(model.nobs),
            'log_likelihood': float(model.llf),
            'pseudo_r_squared': float(model.prsquared),
            'aic': float(model.aic),
            'bic': float(model.bic),
            'converged': bool(model.mle_retvals['converged']) if hasattr(model, 'mle_retvals') else True
        },
        'coefficients': {},
        'confidence_intervals': {},
        'p_values': {},
        'odds_ratios': {},
        'stratified_analysis': {
            'obese_subjects': results.get('obese_n', 0),
            'non_obese_subjects': results.get('non_obese_n', 0),
            'obese_correlation': results.get('obese_correlation', None),
            'non_obese_correlation': results.get('non_obese_correlation', None)
        },
        'interaction_analysis': {
            'interaction_coefficient': results.get('interaction_coef', None),
            'interaction_p_value': results.get('interaction_pvalue', None),
            'interaction_significant': results.get('interaction_pvalue', 1.0) < 0.05
        }
    }
    
    # Extract coefficient information
    for param_name in model.params.index:
        detailed_results['coefficients'][param_name] = float(model.params[param_name])
        detailed_results['p_values'][param_name] = float(model.pvalues[param_name])
        detailed_results['odds_ratios'][param_name] = float(np.exp(model.params[param_name]))
        
        # Confidence intervals
        conf_int = model.conf_int()
        detailed_results['confidence_intervals'][param_name] = {
            'lower': float(conf_int.loc[param_name, 0]),
            'upper': float(conf_int.loc[param_name, 1])
        }
    
    # Save to JSON (with proper serialization)
    json_file = ANALYSIS_DIR / f"statistical_results_{'test' if test_mode else 'full'}.json"
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy_types(detailed_results)
    
    with open(json_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Statistical results JSON saved: {json_file}")
    
    # Create CSV summary table
    csv_data = []
    for param_name in model.params.index:
        csv_data.append({
            'parameter': param_name,
            'coefficient': detailed_results['coefficients'][param_name],
            'odds_ratio': detailed_results['odds_ratios'][param_name],
            'p_value': detailed_results['p_values'][param_name],
            'ci_lower': detailed_results['confidence_intervals'][param_name]['lower'],
            'ci_upper': detailed_results['confidence_intervals'][param_name]['upper'],
            'significant': detailed_results['p_values'][param_name] < 0.05
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_file = ANALYSIS_DIR / f"statistical_results_{'test' if test_mode else 'full'}.csv"
    csv_df.to_csv(csv_file, index=False)
    logger.info(f"Statistical results CSV saved: {csv_file}")
    
    return detailed_results

def create_visualization_plots(df, results, test_mode=False, logger=None):
    """Create comprehensive visualization plots"""
    logger.info("Creating visualization plots...")
    
    # Set up plotting style from config
    plt.style.use(config.get('visualization.style', 'default'))
    sns.set_palette(config.get('visualization.color_palette', 'husl'))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ARDS Obesity-Plateau Pressure Analysis', fontsize=16, fontweight='bold')
    
    # 1. BMI distribution by obesity status
    if 'bmi' in df.columns and 'obese' in df.columns:
        bmi_data = df['bmi'].dropna()
        if len(bmi_data) > 0:
            axes[0, 0].hist(bmi_data, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(30, color='red', linestyle='--', label='BMI 30 (obesity threshold)')
            axes[0, 0].set_xlabel('BMI (kg/m²)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('BMI Distribution')
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, 'No BMI data available', ha='center', va='center')
            axes[0, 0].set_title('BMI Distribution (No Data)')
    
    # 2. Plateau pressure by obesity status
    if 'mean_plateau_pressure' in df.columns and 'obese' in df.columns:
        plateau_data = df[['mean_plateau_pressure', 'obese']].dropna()
        if len(plateau_data) > 0:
            obese_plateau = plateau_data[plateau_data['obese'] == True]['mean_plateau_pressure']
            non_obese_plateau = plateau_data[plateau_data['obese'] == False]['mean_plateau_pressure']
            
            if len(obese_plateau) > 0 and len(non_obese_plateau) > 0:
                axes[0, 1].boxplot([non_obese_plateau, obese_plateau], labels=['Non-obese', 'Obese'])
                axes[0, 1].set_ylabel('Plateau Pressure (cmH2O)')
                axes[0, 1].set_title('Plateau Pressure by Obesity Status')
            else:
                axes[0, 1].text(0.5, 0.5, 'Insufficient data for comparison', ha='center', va='center')
                axes[0, 1].set_title('Plateau Pressure by Obesity (Insufficient Data)')
        else:
            axes[0, 1].text(0.5, 0.5, 'No plateau pressure data', ha='center', va='center')
            axes[0, 1].set_title('Plateau Pressure (No Data)')
    
    # 3. Mortality rates by obesity status
    if 'icu_mortality' in df.columns and 'obese' in df.columns:
        mortality_data = df[['icu_mortality', 'obese']].dropna()
        if len(mortality_data) > 0:
            mortality_by_obesity = mortality_data.groupby('obese')['icu_mortality'].mean()
            if len(mortality_by_obesity) > 0:
                mortality_by_obesity.plot(kind='bar', ax=axes[0, 2])
                axes[0, 2].set_xlabel('Obesity Status')
                axes[0, 2].set_ylabel('ICU Mortality Rate')
                axes[0, 2].set_title('ICU Mortality by Obesity Status')
                axes[0, 2].set_xticklabels(['Non-obese', 'Obese'], rotation=0)
            else:
                axes[0, 2].text(0.5, 0.5, 'No mortality data', ha='center', va='center')
                axes[0, 2].set_title('ICU Mortality (No Data)')
        else:
            axes[0, 2].text(0.5, 0.5, 'No mortality data available', ha='center', va='center')
            axes[0, 2].set_title('ICU Mortality (No Data)')
    
    # 4. Scatter plot: Plateau pressure vs mortality risk
    if 'mean_plateau_pressure' in df.columns and 'icu_mortality' in df.columns:
        scatter_data = df[['mean_plateau_pressure', 'icu_mortality', 'obese']].dropna()
        if len(scatter_data) > 0:
            for obesity_status in [False, True]:
                subset = scatter_data[scatter_data['obese'] == obesity_status]
                if len(subset) > 0:
                    label = 'Obese' if obesity_status else 'Non-obese'
                    axes[1, 0].scatter(subset['mean_plateau_pressure'], subset['icu_mortality'], 
                                     alpha=0.6, label=label)
            
            axes[1, 0].set_xlabel('Plateau Pressure (cmH2O)')
            axes[1, 0].set_ylabel('ICU Mortality')
            axes[1, 0].set_title('Plateau Pressure vs ICU Mortality')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No data for scatter plot', ha='center', va='center')
            axes[1, 0].set_title('Plateau Pressure vs Mortality (No Data)')
    
    # 5. Ventilator-free days by obesity
    if 'ventilator_free_days_28' in df.columns and 'obese' in df.columns:
        vfd_data = df[['ventilator_free_days_28', 'obese']].dropna()
        if len(vfd_data) > 0:
            obese_vfd = vfd_data[vfd_data['obese'] == True]['ventilator_free_days_28']
            non_obese_vfd = vfd_data[vfd_data['obese'] == False]['ventilator_free_days_28']
            
            if len(obese_vfd) > 0 and len(non_obese_vfd) > 0:
                axes[1, 1].boxplot([non_obese_vfd, obese_vfd], labels=['Non-obese', 'Obese'])
                axes[1, 1].set_ylabel('28-day Ventilator-Free Days')
                axes[1, 1].set_title('Ventilator-Free Days by Obesity Status')
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient VFD data', ha='center', va='center')
                axes[1, 1].set_title('Ventilator-Free Days (Insufficient Data)')
        else:
            axes[1, 1].text(0.5, 0.5, 'No VFD data available', ha='center', va='center')
            axes[1, 1].set_title('Ventilator-Free Days (No Data)')
    
    # 6. Model coefficients plot
    if results and 'obesity_coef' in results:
        coef_names = ['Obesity', 'Plateau Pressure', 'Interaction']
        coef_values = [results.get('obesity_coef', 0), 
                      results.get('plateau_coef', 0), 
                      results.get('interaction_coef', 0)]
        
        axes[1, 2].bar(coef_names, coef_values)
        axes[1, 2].set_ylabel('Coefficient Value')
        axes[1, 2].set_title('Logistic Regression Coefficients')
        axes[1, 2].tick_params(axis='x', rotation=45)
    else:
        axes[1, 2].text(0.5, 0.5, 'No model results available', ha='center', va='center')
        axes[1, 2].set_title('Model Coefficients (No Data)')
    
    plt.tight_layout()
    
    # Save figure
    figure_format = config.get('visualization.figure_format', 'png')
    dpi = config.get('visualization.dpi', 300)
    figure_file = FIGURES_DIR / f"comprehensive_analysis_{'test' if test_mode else 'full'}.{figure_format}"
    plt.savefig(figure_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comprehensive analysis figure saved: {figure_file}")
    
    # Create additional statistical visualizations
    create_statistical_visualizations(df, results, test_mode, logger)

def create_statistical_visualizations(df, results, test_mode=False, logger=None):
    """Create additional statistical visualizations"""
    logger.info("Creating enhanced statistical visualizations...")
    
    # Set up plotting parameters
    figure_format = config.get('visualization.figure_format', 'png')
    dpi = config.get('visualization.dpi', 300)
    
    # 1. Interaction Effect Visualization
    if 'interaction_coef' in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Create data for interaction plot
        if 'mean_plateau_pressure' in df.columns and 'obese' in df.columns:
            # Bin plateau pressure for visualization
            df_viz = df.copy()
            df_viz['plateau_bins'] = pd.cut(df_viz['mean_plateau_pressure'], bins=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
            
            # Calculate mortality rates by plateau pressure bins and obesity status
            mortality_by_bins = df_viz.groupby(['plateau_bins', 'obese'])['icu_mortality'].agg(['mean', 'count']).reset_index()
            
            # Plot interaction effect
            for obesity_status in [False, True]:
                subset = mortality_by_bins[mortality_by_bins['obese'] == obesity_status]
                label = 'Obese' if obesity_status else 'Non-obese'
                ax1.plot(subset['plateau_bins'], subset['mean'], marker='o', label=label, linewidth=2)
            
            ax1.set_xlabel('Plateau Pressure (Binned)')
            ax1.set_ylabel('ICU Mortality Rate')
            ax1.set_title('Interaction: Obesity × Plateau Pressure')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Forest plot for coefficients
            coef_names = ['Obesity', 'Plateau Pressure', 'Interaction']
            coef_values = [results.get('obesity_coef', 0), results.get('plateau_coef', 0), results.get('interaction_coef', 0)]
            p_values = [results.get('obesity_pvalue', 1), results.get('plateau_pvalue', 1), results.get('interaction_pvalue', 1)]
            
            # Create forest plot
            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            bars = ax2.barh(coef_names, coef_values, color=colors, alpha=0.7)
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Coefficient Value')
            ax2.set_title('Model Coefficients (Red = p<0.05)')
            ax2.grid(True, alpha=0.3)
            
            # Add p-values as text
            for i, (coef, p_val) in enumerate(zip(coef_values, p_values)):
                ax2.text(coef + 0.01, i, f'p={p_val:.3f}', va='center')
        
        plt.tight_layout()
        interaction_file = FIGURES_DIR / f"interaction_analysis_{'test' if test_mode else 'full'}.{figure_format}"
        plt.savefig(interaction_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Interaction analysis figure saved: {interaction_file}")
    
    # 2. Model Performance Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Model statistics
    if 'model_summary' in results and isinstance(results['model_summary'], dict):
        model_stats = results['model_summary']
        ax1.text(0.5, 0.5, 'Model Performance Metrics\n\n' + 
                 f'N Observations: {model_stats.get("n_observations", "N/A"):,}\n' +
                 f'Log-Likelihood: {model_stats.get("log_likelihood", 0):.3f}\n' +
                 f'Pseudo R²: {model_stats.get("pseudo_r_squared", 0):.3f}\n' +
                 f'AIC: {model_stats.get("aic", 0):.1f}\n' +
                 f'BIC: {model_stats.get("bic", 0):.1f}',
                 ha='center', va='center', fontsize=12, transform=ax1.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax1.set_title('Model Performance Summary')
        ax1.axis('off')
    else:
        ax1.text(0.5, 0.5, 'Model Performance Metrics\n\nModel statistics not available',
                 ha='center', va='center', fontsize=12, transform=ax1.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax1.set_title('Model Performance Summary')
        ax1.axis('off')
    
    # Sample size summary
    sample_info = {
        'Total Subjects': len(df),
        'Obese Subjects': results.get('obese_n', 0),
        'Non-obese Subjects': results.get('non_obese_n', 0),
        'ICU Deaths': df['icu_mortality'].sum() if 'icu_mortality' in df.columns else 0
    }
    
    ax2.text(0.5, 0.5, 'Sample Characteristics\n\n' + 
             '\n'.join([f'{k}: {v:,}' for k, v in sample_info.items()]),
             ha='center', va='center', fontsize=12, transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.set_title('Sample Summary')
    ax2.axis('off')
    
    plt.tight_layout()
    performance_file = FIGURES_DIR / f"model_performance_{'test' if test_mode else 'full'}.{figure_format}"
    plt.savefig(performance_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Model performance figure saved: {performance_file}")

def generate_final_report(df, additional_data, results, test_mode=False, logger=None):
    """Generate final analysis report"""
    logger.info("Generating final analysis report...")
    
    report_file = ANALYSIS_DIR / f"final_analysis_report_{'test' if test_mode else 'full'}.md"
    
    with open(report_file, 'w') as f:
        f.write("# ARDS Obesity-Plateau Pressure Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Mode:** {'Test' if test_mode else 'Full'}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"This report analyzes the interaction between obesity and plateau pressure on clinical outcomes in ARDS patients.\n\n")
        
        # Dataset summary
        f.write("## Dataset Summary\n\n")
        f.write(f"- **Total subjects analyzed:** {len(df)}\n")
        
        if 'obese' in df.columns:
            obese_count = df['obese'].sum()
            f.write(f"- **Obese subjects:** {obese_count} ({obese_count/len(df)*100:.1f}%)\n")
        
        if 'mean_plateau_pressure' in df.columns:
            plateau_subjects = df['mean_plateau_pressure'].notna().sum()
            f.write(f"- **Subjects with plateau pressure data:** {plateau_subjects}\n")
        
        if 'icu_mortality' in df.columns:
            mortality_count = df['icu_mortality'].sum()
            f.write(f"- **ICU mortality:** {mortality_count} ({mortality_count/len(df)*100:.1f}%)\n")
        
        # Key findings
        f.write("\n## Key Findings\n\n")
        
        if results:
            if 'interaction_coef' in results:
                f.write(f"- **Interaction coefficient:** {results['interaction_coef']:.3f}\n")
                f.write(f"- **Interaction p-value:** {results['interaction_pvalue']:.3f}\n")
            
            if 'obese_n' in results and 'non_obese_n' in results:
                f.write(f"- **Obese subjects in analysis:** {results['obese_n']}\n")
                f.write(f"- **Non-obese subjects in analysis:** {results['non_obese_n']}\n")
        
        # Limitations
        f.write("\n## Limitations\n\n")
        f.write("- Small sample size may limit statistical power\n")
        f.write("- Missing data in key variables\n")
        f.write("- Observational study design\n")
        
        # Recommendations
        f.write("\n## Recommendations\n\n")
        f.write("1. Increase sample size for more robust analysis\n")
        f.write("2. Improve data completeness for key variables\n")
        f.write("3. Consider additional confounding variables\n")
        f.write("4. Validate findings in external dataset\n")
        
        # Files generated
        f.write("\n## Generated Files\n\n")
        f.write("- Analysis dataset\n")
        f.write("- Descriptive statistics table\n")
        f.write("- Statistical model results\n")
        f.write("- Comprehensive visualization plots\n")
    
    logger.info(f"Final report saved: {report_file}")
    return report_file

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Statistical Analysis Pipeline')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--full', action='store_true', help='Run full analysis')
    
    args = parser.parse_args()
    
    if not args.test and not args.full:
        print("Please specify either --test or --full mode")
        sys.exit(1)
    
    test_mode = args.test
    
    # Setup logging
    logger = setup_logging(test_mode)
    
    start_time = datetime.now()
    logger.info(f"Statistical analysis started at: {start_time}")
    
    try:
        # Step 1: Load data
        data_result = load_analysis_data(test_mode, logger)
        if data_result is None:
            logger.error("Failed to load analysis data")
            sys.exit(1)
        
        df, additional_data = data_result
        
        # Step 2: Generate descriptive statistics
        table1 = generate_descriptive_statistics(df, additional_data, test_mode, logger)
        
        # Step 3: Analyze obesity-plateau interaction
        results = analyze_obesity_plateau_interaction(df, test_mode, logger)
        
        # Step 4: Create visualizations
        create_visualization_plots(df, results, test_mode, logger)
        
        # Step 5: Generate final report
        report_file = generate_final_report(df, additional_data, results, test_mode, logger)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"Statistical analysis completed successfully!")
        logger.info(f"Duration: {duration}")
        logger.info(f"Subjects analyzed: {len(df)}")
        
        if results:
            logger.info(f"Key findings:")
            if 'interaction_coef' in results:
                logger.info(f"  - Interaction coefficient: {results['interaction_coef']:.3f}")
                logger.info(f"  - Interaction p-value: {results['interaction_pvalue']:.3f}")
        
        logger.info(f"Final report: {report_file}")
        
    except Exception as e:
        logger.error(f"Statistical analysis failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()