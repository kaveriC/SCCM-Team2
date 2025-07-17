# ARDS Detection and Obesity Analysis Project

## 🎯 Research Question
**How does obesity modify the relationship between early plateau pressures and clinical outcomes in ARDS patients, when ARDS onset is accurately detected using unstructured radiology reports?**

## 📊 Project Overview
This project implements advanced NLP techniques (ARDSFlag methodology) to detect ARDS from radiology reports in MIMIC-IV, then analyzes how obesity modifies the relationship between ventilator plateau pressures and patient outcomes.

### Key Features
- **ARDS Detection**: NLP-based bilateral opacity detection from radiology reports
- **Berlin Definition**: Complete implementation of ARDS criteria
- **Obesity Analysis**: WHO-standard BMI classification and interaction analysis
- **Clinical Outcomes**: ICU mortality, ventilator-free days, length of stay
- **Statistical Modeling**: Interaction analysis with multivariable adjustment

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Access to MIMIC-IV datasets (see Data Requirements below)
- 8GB+ RAM recommended
- ~5GB disk space for packages and analysis

### Environment Setup

#### Step 1: Configure Data Paths
**⚠️ IMPORTANT: Complete this step before running any analysis!**

```bash
# Copy the configuration template
cp config.yaml.template config.yaml

# Edit config.yaml to set your MIMIC-IV data paths
vim config.yaml  # or use your preferred editor
```

Update the following paths in `config.yaml`:
```yaml
data:
  base_path: "/path/to/your/MIMIC-IV-3.1/physionet.org/files"
  mimic_note_path: "/path/to/your/MIMIC-IV-3.1/physionet.org/files/mimic-iv-note/2.2"
  mimic_iv_path: "/path/to/your/MIMIC-IV-3.1/physionet.org/files/mimiciv/3.1"
```

#### Step 2: Python Environment Setup

##### Option 1: Automated Setup (Recommended)
```bash
# Clone or navigate to project directory
cd ~/SCCM-Team2

# Run the setup script
./setup_environment.sh

# Activate the environment or select the ards-env kernel
source ards-env/bin/activate

```

##### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv ards-env
source ards-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements (includes PyYAML for config)
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name=ards-env --display-name="ARDS Analysis (Python 3.9)"

# Download spaCy model (optional)
python -m spacy download en_core_web_sm
```

#### Step 3: Validate Configuration
```bash
# Test your configuration
python src/config.py

# You should see a configuration summary with all data files marked as accessible (✓)
```

### Running the Analysis

**Prerequisites:** Ensure you have completed the configuration setup above!

1. **Validate configuration**: Run `python src/config.py` to verify all data files are accessible
2. **Select the correct kernel**: In Jupyter, use kernel "ARDS Analysis (Python 3.9)"

2. **Run notebooks in sequence**:
   - `00_quick_start_demo.ipynb` - Test environment and data access
   - `02_ards_detection_nlp.ipynb` - ARDS detection from radiology
   - `03_ventilator_data_extraction.ipynb` - Extract ventilator parameters
   - `04_obesity_outcomes_extraction.ipynb` - BMI and clinical outcomes
   - `05_statistical_analysis.ipynb` - Final analysis and visualizations

3. **Run on full dataset**
    Test
    - `python src/ards_detection/run_ards_detection.py --test`
    - `python src/data_extraction/run_ventilator_extraction.py --test`
    Full
    - `python src/ards_detection/run_ards_detection.py --full`
    - `python src/data_extraction/run_ventilator_extraction.py --full`
    Terminal
    -  `./run_full_pipeline.sh`

## 📁 Project Structure
```
SCCM-Team2/
├── notebooks/              # Analysis notebooks
│   ├── 00_quick_start_demo.ipynb
│   ├── 02_ards_detection_nlp.ipynb
│   ├── 03_ventilator_data_extraction.ipynb
│   ├── 04_obesity_outcomes_extraction.ipynb
│   └── 05_statistical_analysis.ipynb
├── data/                   # Generated analysis results
├── reference_code/         # ARDSFlag and literature references
├── src/                    # Source code modules (future)
├── requirements.txt        # Python dependencies
├── setup_environment.sh    # Environment setup script
├── CLAUDE.md              # Detailed project specifications
└── README.md              # This file
```

## 📊 Data Requirements

### MIMIC-IV Datasets
Configure data paths in `config.yaml` (see setup instructions above):
```
/path/to/your/MIMIC-IV-3.1/physionet.org/files/
├── mimic-iv-note/         # Clinical notes including radiology
└── mimiciv/               # Main MIMIC-IV database
```

**Note:** The actual data paths are configured in `config.yaml` (not tracked in Git for security).

Required tables:
- `mimic-iv-note`: radiology, radiology_detail
- `mimiciv/hosp`: patients, admissions, labevents
- `mimiciv/icu`: chartevents, icustays, d_items

## 🔬 Methodology

### 1. ARDS Detection (ARDSFlag)
- **Bilateral opacities**: NLP patterns on radiology reports
- **Hypoxemia**: P/F ratio ≤ 300
- **PEEP requirement**: ≥ 5 cmH2O
- **Timing**: Acute onset within 1 week
- **Exclusions**: CHF and other cardiogenic causes

### 2. Obesity Classification (WHO)
- Normal: BMI 18.5–24.9
- Overweight: BMI 25–29.9
- Obese Class I: BMI 30–34.9
- Obese Class II: BMI 35–39.9
- Obese Class III: BMI ≥40

### 3. Statistical Analysis
- Primary outcomes: ICU mortality, 28-day ventilator-free days
- Interaction model: `outcome ~ plateau_pressure * obesity + covariates`
- Sensitivity analyses by obesity class

## 📈 Expected Outputs

Each notebook generates specific outputs in the `data/` folder:
- `bilateral_opacity_detection_results.csv` - NLP results
- `berlin_criteria_assessment.csv` - ARDS cohort
- `obesity_classification.csv` - BMI analysis
- `clinical_outcomes.csv` - Patient outcomes
- `final_analysis_dataset.csv` - Complete dataset
- Statistical models and visualizations

## 🛠️ Troubleshooting

### Common Issues

1. **Kernel Error**: "The kernel failed to start..."
   - Solution: Run `./setup_environment.sh` to create proper environment

2. **Import Errors**: Missing packages
   - Solution: `pip install -r requirements.txt`

3. **Data Not Found**: File path errors
   - Solution: Update data paths in CLAUDE.md or notebook cells

4. **Memory Issues**: Large dataset processing
   - Solution: Use data sampling parameters in notebooks

### Getting Help
- Check `CLAUDE.md` for detailed specifications
- Review notebook markdown cells for step-by-step instructions
- Ensure MIMIC-IV data access is properly configured

## 📚 References

### Methodologies
- **ARDSFlag**: Gandomi et al. (2024) - NLP algorithm for ARDS detection
- **Berlin Definition**: ARDS Definition Task Force (2012)
- **WHO BMI Classification**: WHO Expert Consultation (2004)

### Key Papers
- ARDSFlag implementation papers (see `reference_code/`)
- RESPBert multi-site validation study
- MIMIC-IV database documentation

## 📄 License
This project uses MIMIC-IV data under PhysioNet Credentialed Health Data License.
Code is provided for research purposes.

---
*Last updated: 2025-07-16*