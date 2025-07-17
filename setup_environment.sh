#!/bin/bash
# ARDS Analysis Environment Setup Script

echo "🚀 Setting up ARDS Analysis Environment..."

# Check if we're in the correct directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this script from the project root directory."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment: ards-env"
python3 -m venv ards-env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source ards-env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Install Jupyter kernel
echo "🔧 Installing Jupyter kernel..."
python -m ipykernel install --user --name=ards-env --display-name="ARDS Analysis (Python 3.9)"

# Create data directory
echo "📁 Creating data directory..."
mkdir -p data

# Download spaCy model (optional)
echo "🔤 Downloading spaCy English model (optional for advanced NLP)..."
python -m spacy download en_core_web_sm || echo "⚠️ spaCy model download failed (optional)"

echo "✅ Environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Start Jupyter Lab: jupyter lab"
echo "2. In Jupyter, select kernel: 'ARDS Analysis (Python 3.9)'"
echo "3. Open any of the notebooks in the notebooks/ folder"
echo ""
echo "📚 Available notebooks:"
echo "  - 02_ards_detection_nlp_fixed.ipynb"
echo "  - 03_ventilator_data_extraction.ipynb"
echo "  - 04_obesity_outcomes_extraction.ipynb"
echo "  - 05_statistical_analysis.ipynb"
echo ""
echo "🔧 To activate the environment manually later:"
echo "  source ards-env/bin/activate"