# ECG Sparse Autoencoder Analysis

This project implements a sparse autoencoder for learning interpretable representations of ECG (electrocardiogram) signals from the PTB-XL dataset. The system automatically identifies sparse features that correspond to clinically meaningful patterns and generates human-understandable descriptions using Azure OpenAI.

## Overview

The project consists of several key components:

1. **Data Loader** (`data_loader.py`): Loads and preprocesses PTB-XL ECG dataset
2. **Sparse Autoencoder** (`sparse_autoencoder.py`): PyTorch implementation with L1 sparsity regularization
3. **Training Pipeline** (`train.py`): Complete training workflow with monitoring
4. **Clinical Interpreter** (`clinical_interpreter.py`): Azure OpenAI integration for generating clinical descriptions
5. **Feature Analysis** (`analyze_features.py`): Complete pipeline for finding and interpreting features

## Key Features

- **Sparse Representation Learning**: Uses L1 regularization and optional KL divergence to learn sparse ECG representations
- **Feature Activation Analysis**: Identifies which ECG samples maximally activate specific learned features
- **Clinical Interpretation**: Leverages Azure OpenAI to generate human-readable clinical findings from German ECG reports
- **Comprehensive Reporting**: Generates detailed HTML reports with feature interpretations and statistics

## Dataset

The project uses the PTB-XL dataset, which contains:
- 21,837 clinical 12-lead ECGs (10 seconds)
- Sampling rates: 100Hz and 500Hz
- German clinical reports (e.g., "sinusrhythmus periphere niederspannung")
- Structured diagnostic codes (SCP statements)

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd /home/homesOnMaster/fhauke/ECG
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Azure OpenAI credentials (optional, for clinical interpretation):
```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

## Usage

### 1. Train the Sparse Autoencoder

```bash
python train.py
```

This will:
- Load the PTB-XL dataset
- Train a sparse autoencoder with the following architecture:
  - Input: 12,000 dimensions (12 leads × 1000 time points)
  - Hidden layers: [2048, 1024, 512]
  - Latent dimension: 256 sparse features
  - L1 sparsity regularization
- Save the best model to `checkpoints/best_model.pth`
- Generate training curves and metrics

### 2. Analyze Learned Features

```bash
python analyze_features.py --model_path checkpoints/best_model.pth --top_k 10
```

This will:
- Load the trained model
- Analyze feature activations across the dataset
- Find the top-k most interpretable features
- Generate clinical interpretations using Azure OpenAI
- Create comprehensive HTML and JSON reports

### 3. Test Individual Components

Test the data loader:
```bash
python data_loader.py
```

Test the sparse autoencoder:
```bash
python sparse_autoencoder.py
```

Test the clinical interpreter:
```bash
python clinical_interpreter.py
```

## Model Architecture

### Sparse Autoencoder

The sparse autoencoder uses a bottleneck architecture:

```
Input (12,000) → [2048] → [1024] → [512] → Latent (256) → [512] → [1024] → [2048] → Output (12,000)
```

**Sparsity Regularization:**
- L1 penalty on latent activations: `λ₁ * ||z||₁`
- Optional KL divergence penalty for target sparsity
- Dropout regularization during training

**Loss Function:**
```
L = L_reconstruction + λ₁ * L_L1 + λ_KL * L_KL
```

Where:
- `L_reconstruction = MSE(x, x_reconstructed)`
- `L_L1 = mean(|z|)` (promotes sparsity)
- `L_KL = KL(ρ || ρ_hat)` (optional, maintains target sparsity)

## Feature Interpretation Pipeline

### 1. Feature Activation Analysis
For each learned feature:
- Compute activations across all ECG samples
- Identify samples with highest activation values
- Extract corresponding clinical reports

### 2. Clinical Pattern Recognition
Using Azure OpenAI (GPT-4):
- Analyze German clinical reports
- Identify common medical terminology
- Generate clinical interpretations
- Assess confidence in interpretations

### 3. Report Generation
Creates comprehensive reports including:
- Feature activation statistics
- Clinical interpretations with confidence scores
- Sample ECG reports that activate each feature
- Visual analysis of activation patterns

## Example Results

**Feature #42: Sinus Rhythm Patterns**
- **Confidence**: 85%
- **Clinical Interpretation**: "This feature detects normal sinus rhythm patterns with consistent P-wave morphology and regular R-R intervals."
- **Top Activating Reports**:
  - [0.892] "sinusrhythmus normales ekg"
  - [0.845] "sinusrhythmus periphere niederspannung"
  - [0.823] "regelmäßiger sinusrhythmus"

**Feature #73: Bradycardia Detection**
- **Confidence**: 78%
- **Clinical Interpretation**: "This feature identifies bradycardic rhythms characterized by slow heart rates below 60 bpm."
- **Top Activating Reports**:
  - [0.756] "sinusbradykardie sonst normales ekg"
  - [0.701] "bradykardie bei sinusrhythmus"

## Configuration

### Training Configuration
```python
config = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'sampling_rate': 100,  # Hz
    
    # Model architecture
    'hidden_dims': [2048, 1024, 512],
    'latent_dim': 256,
    'sparsity_weight': 0.01,
    'dropout_rate': 0.2,
}
```

### Azure OpenAI Configuration
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

## Output Files

The system generates several output files:

1. **Model Checkpoints**:
   - `checkpoints/best_model.pth`: Best trained model
   - `checkpoints/training_history.json`: Training metrics
   - `checkpoints/training_curves.png`: Loss curves

2. **Analysis Results**:
   - `feature_interpretations.json`: Structured feature analysis data
   - `feature_analysis_report.html`: Comprehensive HTML report
   - `feature_activation_analysis.png`: Activation visualization

## Technical Details

### Dataset Processing
- ECG signals are loaded using the `wfdb` library
- Signals are normalized to zero mean and unit variance
- 12-lead ECGs are flattened to 12,000-dimensional vectors
- German clinical reports are preprocessed and analyzed

### Sparsity Analysis
- Features are considered "active" if |activation| > threshold (default: 0.1)
- Sparsity metrics track the percentage of near-zero activations
- Feature importance is ranked by mean absolute activation

### Clinical Term Extraction
- German medical terminology is identified using keyword matching
- Common terms are extracted from activating reports
- Medical significance is assessed through pattern analysis

## Limitations and Future Work

### Current Limitations
- Limited to German clinical reports
- Requires Azure OpenAI for best interpretations
- Computational requirements for full dataset processing

### Future Enhancements
- Multi-language support for clinical reports
- Integration with medical ontologies (SNOMED CT, ICD-10)
- Real-time ECG analysis capabilities
- Advanced visualization tools
- Automated clinical decision support

## Research Applications

This system enables several research applications:

1. **Interpretable AI in Cardiology**: Understanding what deep models learn from ECG data
2. **Clinical Feature Discovery**: Identifying novel ECG patterns
3. **Automated Diagnosis Support**: Sparse features as diagnostic markers
4. **Medical Education**: Visual explanation of ECG abnormalities
5. **Quality Assurance**: Detecting artifacts and measurement errors

## Contributing

To extend this project:

1. **Add New Sparsity Techniques**: Implement other sparsity-inducing methods
2. **Enhance Clinical Interpretation**: Integrate medical knowledge bases
3. **Improve Visualization**: Add interactive ECG plotting
4. **Scale to Full Dataset**: Optimize for larger dataset processing
5. **Add Real-time Processing**: Implement streaming ECG analysis

## References

- **PTB-XL Dataset**: Wagner, P., et al. "PTB-XL, a large publicly available electrocardiography dataset." Nature Scientific Data 7.1 (2020): 1-15.
- **Sparse Autoencoders**: Ng, A. "Sparse autoencoder." CS294A Lecture notes 72.2011 (2011): 1-19.
- **ECG Analysis**: Hannun, A.Y., et al. "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." Nature medicine 25.1 (2019): 65-69.

## License

This project is provided for research and educational purposes. Please ensure compliance with PTB-XL dataset licensing terms.
