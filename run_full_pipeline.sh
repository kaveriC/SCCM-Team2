#!/bin/bash
# ARDS Analysis Full Pipeline - Production Run
# This script runs the complete ARDS detection and ventilator data extraction pipeline

echo "🚀 Starting Full ARDS Analysis Pipeline..."
echo "$(date): Pipeline initiated"
echo ""

# Check if we're in the correct directory
if [ ! -f "src/ards_detection/run_ards_detection.py" ] || [ ! -f "src/data_extraction/run_ventilator_extraction.py" ]; then
    echo "❌ Error: Pipeline scripts not found. Please run this script from the project root directory."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "ards-env" ]; then
    echo "🔄 Activating virtual environment..."
    source ards-env/bin/activate
fi

# Create outputs directory if it doesn't exist
mkdir -p outputs/checkpoints outputs/figures outputs/tables

echo "📊 System Information:"
echo "  - Available memory: $(free -h | awk '/^Mem:/ {print $7}' 2>/dev/null || echo 'N/A')"
echo "  - Available disk space: $(df -h . | awk 'NR==2 {print $4}')"
echo "  - CPU cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'N/A')"
echo ""

# Step 1: ARDS Detection Pipeline
echo "📋 Step 1: Running ARDS Detection Pipeline..."
echo "$(date): Starting bilateral opacity detection from radiology reports"
echo "Expected duration: 2-4 hours for full dataset"
echo ""

python src/ards_detection/run_ards_detection.py --full

# Check if Step 1 completed successfully
if [ $? -ne 0 ]; then
    echo "❌ Error: ARDS Detection Pipeline failed!"
    echo "$(date): Pipeline terminated due to Step 1 failure"
    exit 1
fi

echo ""
echo "✅ Step 1 completed successfully!"
echo "$(date): ARDS detection phase completed"
echo ""

# Step 2: Ventilator Data Extraction Pipeline
echo "🫁 Step 2: Running Ventilator Data Extraction Pipeline..."
echo "$(date): Starting ventilator data extraction and Berlin criteria assessment"
echo "Expected duration: 1-2 hours for full dataset"
echo ""

python src/data_extraction/run_ventilator_extraction.py --full

# Check if Step 2 completed successfully
if [ $? -ne 0 ]; then
    echo "❌ Error: Ventilator Data Extraction Pipeline failed!"
    echo "$(date): Pipeline terminated due to Step 2 failure"
    exit 1
fi

echo ""
echo "✅ Step 2 completed successfully!"
echo "$(date): Ventilator data extraction phase completed"
echo ""

# Pipeline completion summary
echo "🎉 Full ARDS Analysis Pipeline Completed Successfully!"
echo "$(date): Pipeline finished"
echo ""
echo "📊 Generated Outputs:"
echo "  - outputs/bilateral_opacity_detection_results_full.csv"
echo "  - outputs/ards_candidates_full.csv"
echo "  - outputs/ards_subject_list_full.pkl"
echo "  - outputs/ventilator_data_full.csv"
echo "  - outputs/berlin_criteria_assessment_full.csv"
echo "  - outputs/figures/ (STROBE diagram, summary plots)"
echo "  - outputs/tables/ (cohort characteristics, statistics)"
echo "  - outputs/checkpoints/ (intermediate data saves)"
echo ""
echo "📈 Next Steps:"
echo "  1. Review log files in outputs/ for detailed processing information"
echo "  2. Examine STROBE diagram for cohort flow"
echo "  3. Proceed with statistical analysis using generated datasets"
echo ""
echo "🔬 Ready for obesity-plateau pressure interaction analysis!"