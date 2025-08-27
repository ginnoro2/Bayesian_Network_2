# Malware Analysis with Machine Learning

**Advanced Machine Learning Assignment - Malware Image Classification using Gaussian Processes, Bayesian Networks, and LDA Topic Modeling**

## What This Project Does

This project analyzes malware images using multiple machine learning algorithms to detect and classify malware families. It implements a complete pipeline from data processing to visualization, addressing real-world cybersecurity challenges.

**Key Features:**
- Comprehensive data analysis and optimization pipeline
- Four different ML algorithms (GP Regression/Classification, Bayesian Networks, LDA)
- Professional visualizations and performance comparison
- Real cybersecurity application with malware detection

## Project Structure

```
task1/
├── data_analysis/                    # Data preprocessing & EDA
│   ├── exploratory_data_analysis.py     # Main EDA script
│   ├── data_optimization.py             # Data optimization pipeline  
│   ├── data_visualization/              # Generated EDA plots
│   ├── data_optimization/               # Optimized datasets & models
│   └── exploratory_analysis/            # EDA reports & processed data
├── code/                             # ML implementations
│   ├── malware_analysis.py              # Main ML algorithms
│   └── comparison_visualizations.py     # Results comparison
├── data/
│   └── malimg_paper_dataset_imgs/    # 25 malware families (1,250 images)
├── results/                            # Model outputs & metrics
├── visualizations/                     # Generated plots & comparisons
├── requirements.txt                    # Python dependencies
└── Documentation files                 # Project guides & summaries
```

## Start With

### 1. Installation

```bash
# Clone or download the project
cd task1

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Analysis

```bash
# Navigate to data analysis folder
cd data_analysis

# Run exploratory data analysis
python exploratory_data_analysis.py

# Run data optimization pipeline
python data_optimization.py
```

**Results:**
- 3 data visualization plots in `data_visualization/`
- Comprehensive analysis report in `exploratory_analysis/`
- 5 optimized datasets in `data_optimization/`

### 3. Run Machine Learning Models

```bash
# Navigate to code folder
cd ../code

# Run main ML analysis (all algorithms)
python malware_analysis.py
```

**Results:**
- Model results saved in `../results/`
- Individual algorithm visualizations in `../visualizations/`
- Summary report with performance metrics

### 4. Generate Comparison Visualizations

```bash
# Generate comprehensive comparison dashboard
python comparison_visualizations.py
```

**Results:**
- `comprehensive_comparison.png` - Complete dashboard
- `performance_summary_table.png` - Tabular comparison

## What Each Script Does

### Data Processing

#### `exploratory_data_analysis.py`
```bash
cd data_analysis
python exploratory_data_analysis.py
```
- **Input**: Raw malware images from 25 families
- **Process**: Extract 30 features per image, analyze distributions
- **Output**: 
  - `data_distribution_analysis.png`
  - `feature_importance_analysis.png`
  - `data_quality_analysis.png`
  - `data_analysis_report.txt`
  - `features_df.pkl`

#### `data_optimization.py`
```bash
python data_optimization.py
```
- **Input**: Processed features from EDA
- **Process**: Outlier removal, scaling, feature selection, PCA, SMOTE balancing
- **Output**:
  - `cleaned_data.csv`
  - `scaled_data.csv` 
  - `selected_features_data.csv`
  - `reduced_dimensions_data.csv`
  - `final_optimized_data.csv`
  - `optimization_evaluation.png`
  - `scaler.pkl`, `pca.pkl`

### Machine Learning

#### `malware_analysis.py`
```bash
cd code
python malware_analysis.py
```
- **Implements 4 ML algorithms:**
  1. **Gaussian Process Regression** (4 inputs → 1 output)
  2. **Gaussian Process Classification** (binary classification)
  3. **Bayesian Network Analysis** (8 variables correlation)
  4. **LDA Topic Modeling** (10 topics from features)

- **Output**:
  - `gpr_results.npy`, `gpc_results.npy`, `bn_results.npy`, `lda_results.npy`
  - `gpr_results.png`, `gpc_results.png`, `bayesian_network_*.png`, `lda_results.png`
  - `summary_report.txt`

#### `comparison_visualizations.py`
```bash
python comparison_visualizations.py
```
- **Creates comprehensive comparison dashboard**
- **Output**:
  - `comprehensive_comparison.png` (complete overview)
  - `performance_summary_table.png` (metrics table)

## Expected Performance Results

| Algorithm | Metric | Expected Value | Status |
|-----------|--------|----------------|--------|
| **Gaussian Process Classification** | Accuracy | ~96.40% 
| **LDA Topic Modeling** | Perplexity | ~43.65 
| **Gaussian Process Regression** | R² Score | ~-11.23 
| **Bayesian Network** | Variables | 8 features | Completed analysis 

## Feature Extraction Details

Each malware image is converted into **30 numerical features**:

1. **Statistical Features (7)**: mean, std, median, min, max, 25th/75th percentiles
2. **Histogram Features (10)**: 10-bin intensity distribution
3. **Texture Features (4)**: Gradient statistics (Sobel operators)
4. **Edge Features (1)**: Edge density using Canny detection
5. **Pattern Features (8)**: Local Binary Pattern (LBP) histogram

## Visualization Outputs

### Data Analysis Visualizations
- `data_distribution_analysis.png` - Family distributions and basic stats
- `feature_importance_analysis.png` - Feature ranking and correlation
- `data_quality_analysis.png` - Missing values, outliers, data quality
- `optimization_evaluation.png` - Before/after optimization comparison

### ML Algorithm Visualizations
- `gpr_results.png` - Regression: actual vs predicted, residuals, uncertainty
- `gpc_results.png` - Classification: confusion matrix, probability distributions
- `bayesian_network_correlation.png` - 8×8 feature correlation heatmap
- `bayesian_network_distributions.png` - Feature distribution histograms
- `lda_results.png` - Topic modeling: topic distributions by malware family

### Comparison Dashboards
- `comprehensive_comparison.png` - Complete 8-panel comparison dashboard
- `performance_summary_table.png` - Clean metrics comparison table

## Saved Results & Models

### Processed Data
- `features_df.pkl` - Original processed features
- `cleaned_data.csv` - Outlier-handled data
- `final_optimized_data.csv` - Complete optimized dataset
- `scaler.pkl`, `pca.pkl` - Saved preprocessing models

### ML Results
- `gpr_results.npy` - Regression predictions, uncertainties, metrics
- `gpc_results.npy` - Classification results, probabilities, confusion matrix
- `bn_results.npy` - Bayesian network correlation matrix, feature stats
- `lda_results.npy` - Topic distributions, perplexity, topic weights
- `summary_report.txt` - Complete performance summary

## Assignment Requirements Fulfilled

### Gaussian Process (GP)
- **4+ Input Variables**: Uses 4 intensity features
- **Single Output**: Predicts max_intensity
- **Binary Classification**: Threshold-based GPC
- **Regression**: Continuous value prediction with uncertainty

### Bayesian Network
- **8+ Random Variables**: Analyzes 8 malware features
- **Hybrid Variables**: Continuous + discretized analysis
- **Correlation Analysis**: Complete correlation matrix
- **Conditional Probabilities**: Feature relationships

### LDA Topic Modeling
- **Topic Discovery**: 10 topics from malware features
- **Pattern Analysis**: Family-specific topic distributions
- **Dimensionality Reduction**: PCA preprocessing
- **No Restrictions**: Applied to full feature set

## Troubleshooting

### Common Issues

**Memory Error during processing:**
```bash
# Reduce max_samples_per_class in scripts
# Edit line in malware_analysis.py: max_samples_per_class=20
```

**Missing data folder:**
```bash
# Ensure data is in correct location:
task1/data/malimg_paper_dataset_imgs/[25 malware folders]/[PNG files]
```

**Import errors:**
```bash
# Ensure virtual environment is activated and dependencies installed
source venv/bin/activate
pip install -r requirements.txt
```

**Visualization not displaying:**
```bash
# For headless environments, plots are saved to files automatically
# Check visualizations/ and data_analysis/data_visualization/ folders
```

## Dependencies

```python
numpy>=1.24.0          # Numerical computations
pandas>=2.0.0          # Data manipulation
matplotlib>=3.7.0      # Plotting
seaborn>=0.12.0        # Statistical visualizations
scikit-learn>=1.3.0    # ML algorithms
scipy>=1.11.0          # Scientific computing
opencv-python>=4.8.0   # Image processing
Pillow>=10.0.0         # Image handling
plotly>=5.15.0         # Interactive plots
tqdm>=4.65.0           # Progress bars
joblib>=1.3.0          # Model persistence
imbalanced-learn>=0.10.0  # Data balancing
```

## Task 1 Objectives 

1. **Comprehensive ML Pipeline**: Complete workflow from raw data to insights
2. **Multiple Algorithm Implementation**: GP, Bayesian Networks, LDA topic modeling
3. **Real-world Application**: Cybersecurity malware detection
4. **Professional Data Science**: EDA, optimization, evaluation, visualization
5. **Academic Requirements**: All assignment criteria fulfilled with documentation

## Key Results Summary

- **Dataset**: 1,250 malware images from 25 families
- **Features**: 30 comprehensive features per image
- **Best Performance**: 96.40% accuracy in malware classification (GPC)
- **Topic Discovery**: 10 meaningful topics identified across malware families
- **Feature Analysis**: 8-variable Bayesian network reveals malware structure
- **Visualizations**: 11+ professional plots and dashboards generated
- **Documentation**: Complete project documentation and guides

---

**Contact**: For questions about implementation details, see documentation files or code comments.
