#!/usr/bin/env python3
"""
Exploratory Data Analysis for Malware Image Dataset
This script performs comprehensive data analysis before applying ML models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MalwareDataExplorer:
    def __init__(self, data_path="../data/malimg_paper_dataset_imgs", 
                 output_path="../data_analysis"):
        self.data_path = data_path
        self.output_path = output_path
        self.features_df = None
        self.labels = None
        self.feature_names = None
        
        # Create output directories
        os.makedirs(os.path.join(output_path, "exploratory_analysis"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "data_visualization"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "data_optimization"), exist_ok=True)
        
    def load_and_analyze_dataset(self, max_samples_per_class=50):
        """Load dataset and perform initial analysis"""
        print("=== EXPLORATORY DATA ANALYSIS ===")
        print("Loading malware image dataset...")
        
        features_list = []
        labels_list = []
        family_info = {}
        
        # Get all malware families
        malware_families = [d for d in os.listdir(self.data_path) 
                          if os.path.isdir(os.path.join(self.data_path, d))]
        
        print(f"Found {len(malware_families)} malware families")
        
        for family in malware_families:
            family_path = os.path.join(self.data_path, family)
            image_files = [f for f in os.listdir(family_path) if f.endswith('.png')]
            
            # Limit samples per class
            image_files = image_files[:max_samples_per_class]
            family_info[family] = len(image_files)
            
            print(f"Processing {family}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(family_path, img_file)
                features = self.extract_image_features(img_path)
                features_list.append(features)
                labels_list.append(family)
        
        # Convert to numpy arrays
        self.features = np.array(features_list)
        self.labels = np.array(labels_list)
        
        # Create feature names
        self.feature_names = [
            'mean_intensity', 'std_intensity', 'median_intensity', 'min_intensity',
            'max_intensity', 'q25_intensity', 'q75_intensity', 'edge_density',
            'texture_energy', 'texture_contrast', 'texture_correlation', 'texture_homogeneity',
            'lbp_hist_1', 'lbp_hist_2', 'lbp_hist_3', 'lbp_hist_4', 'lbp_hist_5',
            'lbp_hist_6', 'lbp_hist_7', 'lbp_hist_8', 'lbp_hist_9', 'lbp_hist_10',
            'haralick_1', 'haralick_2', 'haralick_3', 'haralick_4', 'haralick_5',
            'haralick_6', 'haralick_7', 'haralick_8'
        ]
        
        # Create DataFrame
        self.features_df = pd.DataFrame(self.features, columns=self.feature_names)
        self.features_df['malware_family'] = self.labels
        
        print(f"\nDataset Summary:")
        print(f"Total samples: {len(self.features)}")
        print(f"Feature dimension: {self.features.shape[1]}")
        print(f"Number of classes: {len(malware_families)}")
        
        # Save family info
        self.family_info = family_info
        
        return self.features_df
    
    def extract_image_features(self, image_path):
        """Extract comprehensive features from malware image"""
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros(30)  # Return zeros if image loading fails
            
            # Resize for consistency
            img = cv2.resize(img, (64, 64))
            
            features = []
            
            # 1. Basic intensity features
            features.extend([
                np.mean(img), np.std(img), np.median(img), np.min(img),
                np.max(img), np.percentile(img, 25), np.percentile(img, 75)
            ])
            
            # 2. Edge density
            edges = cv2.Canny(img, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # 3. Texture features (GLCM-like)
            features.extend(self.extract_texture_features(img))
            
            # 4. LBP histogram
            lbp_features = self.extract_lbp_features(img)
            features.extend(lbp_features)
            
            # 5. Haralick-like features
            haralick_features = self.extract_haralick_features(img)
            features.extend(haralick_features)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return np.zeros(30)
    
    def extract_texture_features(self, img):
        """Extract texture features"""
        # Simple texture measures
        energy = np.sum(img**2) / (img.shape[0] * img.shape[1])
        contrast = np.std(img)
        correlation = np.corrcoef(img.flatten(), np.arange(len(img.flatten())))[0, 1]
        homogeneity = np.sum(img) / (img.shape[0] * img.shape[1])
        
        return [energy, contrast, correlation, homogeneity]
    
    def extract_lbp_features(self, img):
        """Extract Local Binary Pattern features"""
        # Simplified LBP
        lbp = np.zeros_like(img)
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                center = img[i, j]
                code = 0
                code |= (img[i-1, j-1] > center) << 7
                code |= (img[i-1, j] > center) << 6
                code |= (img[i-1, j+1] > center) << 5
                code |= (img[i, j+1] > center) << 4
                code |= (img[i+1, j+1] > center) << 3
                code |= (img[i+1, j] > center) << 2
                code |= (img[i+1, j-1] > center) << 1
                code |= (img[i, j-1] > center) << 0
                lbp[i, j] = code
        
        # Calculate histogram
        hist, _ = np.histogram(lbp, bins=10, range=(0, 256))
        return hist.tolist()
    
    def extract_haralick_features(self, img):
        """Extract Haralick-like texture features"""
        # Simplified Haralick features
        features = []
        
        # Angular second moment
        features.append(np.sum(img**2) / (img.shape[0] * img.shape[1]))
        
        # Contrast
        features.append(np.var(img))
        
        # Correlation
        features.append(np.corrcoef(img.flatten(), np.arange(len(img.flatten())))[0, 1])
        
        # Variance
        features.append(np.var(img))
        
        # Inverse difference moment
        features.append(np.sum(1 / (1 + img**2)) / (img.shape[0] * img.shape[1]))
        
        # Sum average
        features.append(np.mean(img))
        
        # Sum variance
        features.append(np.var(img))
        
        # Sum entropy
        hist, _ = np.histogram(img, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features.append(entropy)
        
        return features
    
    def analyze_data_distribution(self):
        """Analyze data distribution across malware families"""
        print("\n=== DATA DISTRIBUTION ANALYSIS ===")
        
        # Family distribution
        family_counts = Counter(self.labels)
        
        # Create distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Family distribution
        families = list(family_counts.keys())
        counts = list(family_counts.values())
        
        axes[0, 0].bar(range(len(families)), counts, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Malware Family Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Malware Family')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Feature distribution (first 8 features)
        feature_data = self.features_df.iloc[:, :8]
        feature_data.boxplot(ax=axes[0, 1])
        axes[0, 1].set_title('Feature Distribution (First 8 Features)', fontsize=14, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Correlation heatmap
        correlation_matrix = feature_data.corr()
        im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(range(len(feature_data.columns)))
        axes[1, 0].set_yticks(range(len(feature_data.columns)))
        axes[1, 0].set_xticklabels(feature_data.columns, rotation=45, ha='right')
        axes[1, 0].set_yticklabels(feature_data.columns)
        
        # Add correlation values
        for i in range(len(feature_data.columns)):
            for j in range(len(feature_data.columns)):
                text = axes[1, 0].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=10)
        
        # 4. Feature statistics
        stats_data = feature_data.describe()
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=stats_data.values, 
                                colLabels=stats_data.columns,
                                rowLabels=stats_data.index,
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Feature Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'data_visualization', 'data_distribution_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Data distribution analysis saved")
        
        return family_counts, correlation_matrix
    
    def analyze_feature_importance(self):
        """Analyze feature importance using various methods"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Prepare data
        X = self.features_df.iloc[:, :-1]  # All features except label
        y = self.features_df['malware_family']
        
        # 1. Variance analysis
        feature_variance = X.var().sort_values(ascending=False)
        
        # 2. Correlation with target (using label encoding)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        correlations = []
        for col in X.columns:
            corr = np.corrcoef(X[col], y_encoded)[0, 1]
            correlations.append(abs(corr))
        
        feature_correlation = pd.Series(correlations, index=X.columns).sort_values(ascending=False)
        
        # 3. Mutual information
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
        feature_mi = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Variance-based importance
        top_variance = feature_variance.head(10)
        axes[0, 0].barh(range(len(top_variance)), top_variance.values, color='lightcoral')
        axes[0, 0].set_yticks(range(len(top_variance)))
        axes[0, 0].set_yticklabels([f.replace('_', '\n') for f in top_variance.index])
        axes[0, 0].set_title('Feature Importance (Variance)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Variance')
        
        # 2. Correlation-based importance
        top_correlation = feature_correlation.head(10)
        axes[0, 1].barh(range(len(top_correlation)), top_correlation.values, color='lightblue')
        axes[0, 1].set_yticks(range(len(top_correlation)))
        axes[0, 1].set_yticklabels([f.replace('_', '\n') for f in top_correlation.index])
        axes[0, 1].set_title('Feature Importance (Correlation)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Absolute Correlation')
        
        # 3. Mutual information importance
        top_mi = feature_mi.head(10)
        axes[1, 0].barh(range(len(top_mi)), top_mi.values, color='lightgreen')
        axes[1, 0].set_yticks(range(len(top_mi)))
        axes[1, 0].set_yticklabels([f.replace('_', '\n') for f in top_mi.index])
        axes[1, 0].set_title('Feature Importance (Mutual Information)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Mutual Information')
        
        # 4. Combined importance score
        # Normalize and combine scores
        norm_variance = (feature_variance - feature_variance.min()) / (feature_variance.max() - feature_variance.min())
        norm_correlation = (feature_correlation - feature_correlation.min()) / (feature_correlation.max() - feature_correlation.min())
        norm_mi = (feature_mi - feature_mi.min()) / (feature_mi.max() - feature_mi.min())
        
        combined_score = (norm_variance + norm_correlation + norm_mi) / 3
        top_combined = combined_score.sort_values(ascending=False).head(10)
        
        axes[1, 1].barh(range(len(top_combined)), top_combined.values, color='gold')
        axes[1, 1].set_yticks(range(len(top_combined)))
        axes[1, 1].set_yticklabels([f.replace('_', '\n') for f in top_combined.index])
        axes[1, 1].set_title('Feature Importance (Combined Score)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Combined Importance Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'data_visualization', 'feature_importance_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Feature importance analysis saved")
        
        return feature_variance, feature_correlation, feature_mi, combined_score
    
    def analyze_data_quality(self):
        """Analyze data quality issues"""
        print("\n=== DATA QUALITY ANALYSIS ===")
        
        # Check for missing values
        missing_values = self.features_df.isnull().sum()
        
        # Check for duplicate rows
        duplicates = self.features_df.duplicated().sum()
        
        # Check for constant features
        constant_features = []
        for col in self.features_df.columns[:-1]:  # Exclude label
            if self.features_df[col].nunique() == 1:
                constant_features.append(col)
        
        # Check for outliers using IQR method
        outlier_counts = {}
        for col in self.features_df.columns[:-1]:
            Q1 = self.features_df[col].quantile(0.25)
            Q3 = self.features_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((self.features_df[col] < lower_bound) | 
                       (self.features_df[col] > upper_bound)).sum()
            outlier_counts[col] = outliers
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Missing values
        missing_plot = missing_values[missing_values > 0]
        if len(missing_plot) > 0:
            axes[0, 0].bar(range(len(missing_plot)), missing_plot.values, color='red', alpha=0.7)
            axes[0, 0].set_xticks(range(len(missing_plot)))
            axes[0, 0].set_xticklabels(missing_plot.index, rotation=45)
            axes[0, 0].set_title('Missing Values', fontsize=14, fontweight='bold')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                           transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold')
            axes[0, 0].set_title('Missing Values', fontsize=14, fontweight='bold')
        
        # 2. Outlier analysis
        outlier_series = pd.Series(outlier_counts)
        top_outliers = outlier_series.sort_values(ascending=False).head(10)
        axes[0, 1].bar(range(len(top_outliers)), top_outliers.values, color='orange', alpha=0.7)
        axes[0, 1].set_xticks(range(len(top_outliers)))
        axes[0, 1].set_xticklabels([f.replace('_', '\n') for f in top_outliers.index], rotation=45)
        axes[0, 1].set_title('Outlier Count (Top 10 Features)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Number of Outliers')
        
        # 3. Data quality summary
        quality_data = {
            'Metric': ['Total Samples', 'Total Features', 'Missing Values', 'Duplicate Rows', 'Constant Features'],
            'Value': [len(self.features_df), len(self.features_df.columns)-1, 
                     missing_values.sum(), duplicates, len(constant_features)]
        }
        quality_df = pd.DataFrame(quality_data)
        
        axes[1, 0].axis('tight')
        axes[1, 0].axis('off')
        table = axes[1, 0].table(cellText=quality_df.values, 
                                colLabels=quality_df.columns,
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        axes[1, 0].set_title('Data Quality Summary', fontsize=14, fontweight='bold', pad=20)
        
        # 4. Feature value ranges
        feature_ranges = self.features_df.iloc[:, :-1].describe()
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table2 = axes[1, 1].table(cellText=feature_ranges.values, 
                                 colLabels=feature_ranges.columns,
                                 rowLabels=feature_ranges.index,
                                 cellLoc='center', loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(8)
        table2.scale(1.2, 1.5)
        axes[1, 1].set_title('Feature Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'data_visualization', 'data_quality_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Data quality analysis saved")
        
        return {
            'missing_values': missing_values,
            'duplicates': duplicates,
            'constant_features': constant_features,
            'outlier_counts': outlier_counts
        }
    
    def generate_data_analysis_report(self):
        """Generate comprehensive data analysis report"""
        print("\n=== GENERATING DATA ANALYSIS REPORT ===")
        
        # Perform all analyses
        family_counts, correlation_matrix = self.analyze_data_distribution()
        feature_variance, feature_correlation, feature_mi, combined_score = self.analyze_feature_importance()
        quality_issues = self.analyze_data_quality()
        
        # Generate report
        report = f"""
============================================================
MALWARE DATASET - EXPLORATORY DATA ANALYSIS REPORT
============================================================

1. DATASET OVERVIEW
-------------------
Total Samples: {len(self.features_df)}
Total Features: {len(self.features_df.columns)-1}
Number of Classes: {len(family_counts)}
Dataset Shape: {self.features_df.shape}

2. MALWARE FAMILY DISTRIBUTION
------------------------------
{chr(10).join([f"{family}: {count} samples" for family, count in family_counts.items()])}

3. FEATURE IMPORTANCE ANALYSIS
-----------------------------
Top 5 Features by Variance:
{chr(10).join([f"{i+1}. {feature}: {value:.4f}" for i, (feature, value) in enumerate(feature_variance.head(5).items())])}

Top 5 Features by Correlation:
{chr(10).join([f"{i+1}. {feature}: {value:.4f}" for i, (feature, value) in enumerate(feature_correlation.head(5).items())])}

Top 5 Features by Mutual Information:
{chr(10).join([f"{i+1}. {feature}: {value:.4f}" for i, (feature, value) in enumerate(feature_mi.head(5).items())])}

4. DATA QUALITY ASSESSMENT
-------------------------
Missing Values: {quality_issues['missing_values'].sum()}
Duplicate Rows: {quality_issues['duplicates']}
Constant Features: {len(quality_issues['constant_features'])}
Features with Outliers: {sum(1 for count in quality_issues['outlier_counts'].values() if count > 0)}

5. RECOMMENDATIONS FOR DATA OPTIMIZATION
---------------------------------------
- Feature Selection: Consider using top features based on combined importance score
- Outlier Handling: {sum(1 for count in quality_issues['outlier_counts'].values() if count > 0)} features have outliers
- Data Balancing: Family distribution shows some imbalance
- Feature Scaling: Features have different scales, normalization recommended

6. CORRELATION ANALYSIS
----------------------
Strong Correlations (>0.7):
"""
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append((correlation_matrix.columns[i], 
                                             correlation_matrix.columns[j], corr_value))
        
        for feat1, feat2, corr in strong_correlations[:10]:  # Top 10
            report += f"- {feat1} <-> {feat2}: {corr:.3f}\n"
        
        # Save report
        with open(os.path.join(self.output_path, 'exploratory_analysis', 'data_analysis_report.txt'), 'w') as f:
            f.write(report)
        
        print("Data analysis report saved")
        
        return report

def main():
    """Main function to run complete data analysis"""
    explorer = MalwareDataExplorer()
    
    # Load and analyze dataset
    features_df = explorer.load_and_analyze_dataset(max_samples_per_class=50)
    
    # Generate comprehensive analysis
    report = explorer.generate_data_analysis_report()
    
    print("\n=== DATA ANALYSIS COMPLETE ===")
    print("Files generated:")
    print("- data_distribution_analysis.png")
    print("- feature_importance_analysis.png") 
    print("- data_quality_analysis.png")
    print("- data_analysis_report.txt")
    
    return features_df

if __name__ == "__main__":
    main() 