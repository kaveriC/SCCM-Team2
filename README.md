# ARDS Obesity Study - SCCM Datathon

## Research Question
How does obesity modify the relationship between early plateau pressures and clinical outcomes in ARDS patients, when ARDS onset is accurately detected using unstructured radiology reports?

## Project Structure
```
SCCM-TEAM2/
├── src/                      # Source code
│   ├── data_processing/      # Data loading and preprocessing
│   ├── ards_detection/       # ARDS identification algorithms
│   │   ├── structured/       # Using structured data (ICD codes, etc.)
│   │   └── unstructured/     # Using radiology reports (NLP)
│   ├── analysis/             # Statistical analysis
│   ├── visualization/        # Plotting and figures
│   └── utils/                # Helper functions
├── notebooks/                # Jupyter notebooks for exploration
├── data/                     # Local data storage (gitignored)
├── configs/                  # Configuration files
├── tests/                    # Unit tests
├── docs/                     # Documentation
└── results/                  # Output files
    ├── figures/              # Generated plots
    ├── tables/               # Result tables
    └── models/               # Saved models
```

## Data Sources
- MIMIC-IV v3.1: Clinical data
- MIMIC-CXR: Chest X-ray reports
- MIMIC Notes

## Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
See notebooks/ directory for analysis workflows.

## Team
SCCM Datathon participants

## License
This project uses MIMIC data under PhysioNet Credentialed Health Data License.