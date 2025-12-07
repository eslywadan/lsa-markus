#!/bin/bash
# Setup and test script for LSA-Markus pipeline

echo "==================================="
echo "LSA-Markus Pipeline Setup & Test"
echo "==================================="
echo ""

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Install dependencies
echo "1. Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "2. Checking Kedro installation..."
kedro --version

echo ""
echo "3. Checking project structure..."
kedro info

echo ""
echo "4. Listing available pipelines..."
kedro registry list

echo ""
echo "5. Visualizing pipeline structure (optional - opens browser)..."
echo "   Run 'kedro viz' to see the pipeline visualization"

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "  1. Review parameters in conf/base/parameters.yml"
echo "  2. Run the pipeline:"
echo "     - Full pipeline: kedro run"
echo "     - Data processing only: kedro run --pipeline=data_processing"
echo "     - LSA only: kedro run --pipeline=lsa"
echo "     - LDA only: kedro run --pipeline=lda"
echo ""
echo "  3. View results in data/08_reporting/"
echo ""
