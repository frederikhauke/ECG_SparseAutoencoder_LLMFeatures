#!/bin/bash

# ECG Sparse Autoencoder Setup Script

echo "Setting up ECG Sparse Autoencoder project..."

# Create necessary directories
mkdir -p checkpoints
mkdir -p outputs
mkdir -p logs

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if PTB-XL dataset exists
if [ ! -d "physionet.org/files/ptb-xl/1.0.3/" ]; then
    echo "Warning: PTB-XL dataset not found in expected location."
    echo "Please ensure the dataset is available at: physionet.org/files/ptb-xl/1.0.3/"
    echo "You can download it from: https://physionet.org/content/ptb-xl/1.0.3/"
fi

# Copy environment template if .env doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env file. Please edit it with your Azure OpenAI credentials."
fi

echo "Setup completed!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your Azure OpenAI credentials (optional)"
echo "2. Run: python train.py (to train the model)"
echo "3. Run: python analyze_features.py (to analyze features)"
echo ""
echo "For a quick demo with limited data:"
echo "python demo.py"
