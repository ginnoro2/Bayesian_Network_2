#!/usr/bin/env python3
"""
Comparison Visualizations for Malware Analysis Results
This script creates comprehensive comparison plots for all ML algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComparisonVisualizer:
    def __init__(self, results_path="../results", viz_path="../visualizations"):
        self.results_path = results_path
        self.viz_path = viz_path
        
    def load_results(self):
        """Load all analysis results"""
        print("Loading analysis results...")
        
        # Load GPR results
        gpr_results = np.load(os.path.join(self.results_path, 'gpr_results.npy'), allow_pickle=True).item()
        
        # Load GPC results
        gpc_results = np.load(os.path.join(self.results_path, 'gpc_results.npy'), allow_pickle=True).item()
        
        # Load Bayesian Network results
        bn_results = np.load(os.path.join(self.results_path, 'bn_results.npy'), allow_pickle=True).item()
        
        # Load LDA results
        lda_results = np.load(os.path.join(self.results_path, 'lda_results.npy'), allow_pickle=True).item()
        
        return gpr_results, gpc_results, bn_results, lda_results
    
    def create_comprehensive_comparison(self):
        """Create comprehensive comparison visualization"""
        print("Creating comprehensive comparison visualization...")
        
        # Load results
        gpr_results, gpc_results, bn_results, lda_results = self.load_results()
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1])
        
        # 1. Algorithm Performance Comparison
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_algorithm_performance(ax1, gpr_results, gpc_results, lda_results)
        
        # 2. GPR vs GPC Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_gpr_gpc_comparison(ax2, gpr_results, gpc_results)
        
        # 3. Feature Importance (from Bayesian Network)
        ax3 = fig.add_subplot(gs[1, 1])
        self.plot_feature_importance(ax3, bn_results)
        
        # 4. Topic Distribution (LDA)
        ax4 = fig.add_subplot(gs[1, 2])
        self.plot_topic_distribution(ax4, lda_results)
        
        # 5. Correlation Matrix (Bayesian Network)
        ax5 = fig.add_subplot(gs[2, 0])
        self.plot_correlation_matrix(ax5, bn_results)
        
        # 6. Prediction Confidence (GPC)
        ax6 = fig.add_subplot(gs[2, 1])
        self.plot_prediction_confidence(ax6, gpc_results)
        
        # 7. Residual Analysis (GPR)
        ax7 = fig.add_subplot(gs[2, 2])
        self.plot_residual_analysis(ax7, gpr_results)
        
        # 8. Malware Family Analysis (LDA)
        ax8 = fig.add_subplot(gs[3, :])
        self.plot_malware_family_analysis(ax8, lda_results)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_path, 'comprehensive_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Comprehensive comparison saved as 'comprehensive_comparison.png'")
    
    def plot_algorithm_performance(self, ax, gpr_results, gpc_results, lda_results):
        """Plot overall algorithm performance comparison"""
        algorithms = ['Gaussian Process\nRegression', 'Gaussian Process\nClassification', 'LDA Topic\nModeling']
        
        # Performance metrics
        gpr_score = abs(gpr_results['r2'])  # Use absolute value for visualization
        gpc_score = gpc_results['accuracy']
        lda_score = 1 / lda_results['perplexity'] * 100  # Convert perplexity to score
        
        scores = [gpr_score, gpc_score, lda_score]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = ax.bar(algorithms, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Performance Score', fontsize=12)
        ax.set_ylim(0, max(scores) * 1.2)
        ax.grid(True, alpha=0.3)
        
        # Add performance descriptions
        descriptions = [
            f'R² Score: {gpr_results["r2"]:.3f}',
            f'Accuracy: {gpc_results["accuracy"]:.3f}',
            f'Perplexity: {lda_results["perplexity"]:.1f}'
        ]
        
        for i, desc in enumerate(descriptions):
            ax.text(i, scores[i] + 0.02, desc, ha='center', va='bottom', 
                   fontsize=10, style='italic')
    
    def plot_gpr_gpc_comparison(self, ax, gpr_results, gpc_results):
        """Compare GPR and GPC performance"""
        metrics = ['MSE', 'RMSE', 'R²', 'Accuracy']
        gpr_values = [gpr_results['mse'], gpr_results['rmse'], gpr_results['r2'], 0]
        gpc_values = [0, 0, 0, gpc_results['accuracy']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gpr_values, width, label='GPR', alpha=0.8, color='#FF6B6B')
        bars2 = ax.bar(x + width/2, gpc_values, width, label='GPC', alpha=0.8, color='#4ECDC4')
        
        ax.set_title('GPR vs GPC Performance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_feature_importance(self, ax, bn_results):
        """Plot feature importance based on correlation strength"""
        correlation_matrix = bn_results['correlation_matrix']
        feature_names = bn_results['feature_names']
        
        # Calculate average absolute correlation for each feature
        avg_correlations = np.mean(np.abs(correlation_matrix), axis=1)
        
        # Sort features by importance
        sorted_indices = np.argsort(avg_correlations)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_correlations = avg_correlations[sorted_indices]
        
        bars = ax.barh(range(len(sorted_features)), sorted_correlations, 
                      color='#FFA07A', alpha=0.8, edgecolor='black')
        
        ax.set_title('Feature Importance\n(Based on Correlation)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Average Absolute Correlation', fontsize=12)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels([f.replace('_', '\n') for f in sorted_features])
        ax.grid(True, alpha=0.3)
    
    def plot_topic_distribution(self, ax, lda_results):
        """Plot LDA topic distribution"""
        topic_distributions = lda_results['topic_distributions']
        n_topics = lda_results['n_topics']
        
        # Calculate average topic weights
        avg_topic_weights = np.mean(topic_distributions, axis=0)
        
        bars = ax.bar(range(1, n_topics + 1), avg_topic_weights, 
                     color='#45B7D1', alpha=0.8, edgecolor='black')
        
        ax.set_title('Average Topic Weights\n(LDA)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Topic Number', fontsize=12)
        ax.set_ylabel('Average Weight', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def plot_correlation_matrix(self, ax, bn_results):
        """Plot feature correlation matrix"""
        correlation_matrix = bn_results['correlation_matrix']
        feature_names = bn_results['feature_names']
        
        # Create shortened feature names for better display
        short_names = [name.replace('_intensity', '').replace('_', '\n') for name in feature_names]
        
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Feature Correlation Matrix\n(Bayesian Network)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right')
        ax.set_yticklabels(short_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=10)
    
    def plot_prediction_confidence(self, ax, gpc_results):
        """Plot GPC prediction confidence distribution"""
        probabilities = gpc_results['probabilities']
        confidence = np.max(probabilities, axis=1)
        
        ax.hist(confidence, bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black')
        ax.set_title('GPC Prediction Confidence\nDistribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add mean confidence line
        mean_conf = np.mean(confidence)
        ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_conf:.3f}')
        ax.legend()
    
    def plot_residual_analysis(self, ax, gpr_results):
        """Plot GPR residual analysis"""
        actual = gpr_results['actual']
        predictions = gpr_results['predictions']
        residuals = actual - predictions
        
        ax.scatter(predictions, residuals, alpha=0.6, color='#FF6B6B')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_title('GPR Residual Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def plot_malware_family_analysis(self, ax, lda_results):
        """Plot malware family analysis using LDA topics"""
        family_topic_analysis = lda_results['family_topic_analysis']
        n_topics = lda_results['n_topics']
        
        # Get family names and their topic distributions
        family_names = list(family_topic_analysis.keys())
        topic_means = np.array([family_topic_analysis[family]['mean_distribution'] 
                              for family in family_names])
        
        # Create heatmap
        im = ax.imshow(topic_means, cmap='viridis', aspect='auto')
        
        # Set labels
        ax.set_title('Malware Family Topic Distribution\n(LDA Analysis)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Topic Number', fontsize=12)
        ax.set_ylabel('Malware Family', fontsize=12)
        
        # Set tick labels
        ax.set_xticks(range(n_topics))
        ax.set_xticklabels([f'T{i+1}' for i in range(n_topics)])
        ax.set_yticks(range(len(family_names)))
        ax.set_yticklabels(family_names, fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Topic Weight', fontsize=10)
    
    def create_performance_summary(self):
        """Create a performance summary table visualization"""
        print("Creating performance summary table...")
        
        # Load results
        gpr_results, gpc_results, bn_results, lda_results = self.load_results()
        
        # Create summary data
        summary_data = {
            'Algorithm': ['Gaussian Process Regression', 'Gaussian Process Classification', 
                         'Bayesian Network', 'Latent Dirichlet Allocation'],
            'Primary Metric': ['R² Score', 'Accuracy', 'Variables', 'Perplexity'],
            'Value': [f"{gpr_results['r2']:.4f}", f"{gpc_results['accuracy']:.4f}", 
                     f"{len(bn_results['feature_names'])}", f"{lda_results['perplexity']:.2f}"],
            'Secondary Metric': ['MSE', 'Precision', 'Correlations', 'Topics'],
            'Secondary Value': [f"{gpr_results['mse']:.2f}", f"{gpc_results['accuracy']:.4f}", 
                              f"{len(bn_results['feature_names'])}²", f"{lda_results['n_topics']}"]
        }
        
        df = pd.DataFrame(summary_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F7F7F7')
        
        plt.title('Machine Learning Algorithm Performance Summary', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.savefig(os.path.join(self.viz_path, 'performance_summary_table.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Performance summary table saved as 'performance_summary_table.png'")

def main():
    """Main function to create all comparison visualizations"""
    visualizer = ComparisonVisualizer()
    
    # Create comprehensive comparison
    visualizer.create_comprehensive_comparison()
    
    # Create performance summary table
    visualizer.create_performance_summary()
    
    print("\nAll comparison visualizations created successfully!")
    print("Files generated:")
    print("- comprehensive_comparison.png")
    print("- performance_summary_table.png")

if __name__ == "__main__":
    main() 