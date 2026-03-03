"""
Script to load and analyze saved embeddings.

Loads embeddings from .npz files and provides visualization and analysis tools:
- Load embeddings from folders
- Compute statistics (mean, std, max, min)
- Draw Gaussian distributions for different dimensions
- Visualize 2D/3D embeddings
- Generate histograms and violin plots
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class EmbeddingAnalyzer:
    """Analyze embeddings saved during validation."""
    
    def __init__(self, embeddings_dir: Path):
        """
        Initialize analyzer with path to embeddings directory.
        
        Args:
            embeddings_dir: Path to embeddings folder (e.g., outputs/2026-02-17/20-53-16/embeddings/)
        """
        self.embeddings_dir = Path(embeddings_dir)
        assert self.embeddings_dir.exists(), f"Embeddings directory not found: {embeddings_dir}"
        
        # Check if it's readable and has content
        try:
            contents = list(self.embeddings_dir.iterdir())
            if not contents:
                print(f"⚠️  WARNING: Embeddings directory is empty: {self.embeddings_dir}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading: {self.embeddings_dir}")
        
        self.embeddings = {}
        self.stats = {}
        self._load_embeddings()
    
    def _load_embeddings(self) -> None:
        """Load all embeddings from .npz files."""
        print(f"Loading embeddings from: {self.embeddings_dir}\n")
        
        # Find all validation_step folders
        step_folders = sorted([
            d for d in self.embeddings_dir.iterdir() 
            if d.is_dir() and 'step' in d.name
        ])
        
        print(f"Found {len(step_folders)} validation steps")
        
        if not step_folders:
            print("⚠️  No validation_step_* folders found!")
            print(f"Contents of {self.embeddings_dir}:")
            for item in self.embeddings_dir.iterdir():
                print(f"  {item}")
            return
        
        total_files = 0
        
        for step_folder in step_folders:
            step_name = step_folder.name
            self.embeddings[step_name] = {}
            
            # Find all batch folders
            batch_folders = sorted([
                d for d in step_folder.iterdir() 
                if d.is_dir() and 'batch' in d.name
            ])
            
            print(f"\n  📁 {step_name}: ", end="")
            
            if not batch_folders:
                print("❌ No batch folders found")
                print(f"     Contents of {step_folder}:")
                for item in step_folder.iterdir():
                    if item.is_dir():
                        print(f"       📁 {item.name}/")
                        for sub_item in item.iterdir():
                            print(f"          - {sub_item.name}")
                    else:
                        print(f"       📄 {item.name}")
                continue
            
            print(f"({len(batch_folders)} batches)")
            
            for batch_folder in batch_folders:
                batch_name = batch_folder.name
                self.embeddings[step_name][batch_name] = {}
                
                # Load all .npz files in this batch
                npz_files = list(batch_folder.glob('*.npz'))
                
                if not npz_files:
                    print(f"    ⚠️  {batch_name}: No .npz files")
                    continue
                
                for npz_file in npz_files:
                    embedding_type = npz_file.stem  # e.g., 'gt', 'prediction'
                    try:
                        data = np.load(npz_file)
                        embedding = data['data']
                        self.embeddings[step_name][batch_name][embedding_type] = embedding
                        shape_str = str(embedding.shape)
                        print(f"    ✓ {batch_name}/{embedding_type:12s}: {shape_str}")
                        total_files += 1
                    except Exception as e:
                        print(f"    ❌ {batch_name}/{embedding_type:12s}: {str(e)}")
        
        print(f"\n✅ Loaded {total_files} embedding files\n")
    
    def get_stats(self, step: Optional[str] = None, batch: Optional[str] = None, 
                   embedding_type: str = "gt") -> Dict:
        """
        Compute statistics for embeddings.
        
        Args:
            step: Validation step name (e.g., 'validation_step_10'). If None, uses first step.
            batch: Batch name (e.g., 'batch_000'). If None, uses first batch.
            embedding_type: Type of embedding to analyze (e.g., 'gt', 'prediction')
        
        Returns:
            Dictionary with statistics
        """
        if step is None:
            step = list(self.embeddings.keys())[0]
        if batch is None:
            batch = list(self.embeddings[step].keys())[0]
        
        embedding = self.embeddings[step][batch][embedding_type]
        
        # Flatten and reshape
        flattened = embedding.reshape(-1)
        
        stats_dict = {
            'shape': embedding.shape,
            'mean': float(np.mean(flattened)),
            'std': float(np.std(flattened)),
            'min': float(np.min(flattened)),
            'max': float(np.max(flattened)),
            'median': float(np.median(flattened)),
            'q25': float(np.percentile(flattened, 25)),
            'q75': float(np.percentile(flattened, 75)),
        }
        return stats_dict
    
    def print_stats(self, step: Optional[str] = None, batch: Optional[str] = None,
                    embedding_type: str = "gt") -> None:
        """Print statistics for embeddings."""
        # Validate embeddings exist
        if not self.embeddings:
            print("\n❌ No embeddings loaded!")
            return
        
        if step is None:
            step = list(self.embeddings.keys())[0]
        
        if step not in self.embeddings:
            print(f"\n❌ Step '{step}' not found. Available: {list(self.embeddings.keys())}")
            return
        
        if batch is None:
            batches = list(self.embeddings[step].keys())
            if not batches:
                print(f"\n❌ No batches found in {step}")
                return
            batch = batches[0]
        
        if batch not in self.embeddings[step]:
            print(f"\n❌ Batch '{batch}' not found in {step}")
            print(f"   Available: {list(self.embeddings[step].keys())}")
            return
        
        if embedding_type not in self.embeddings[step][batch]:
            print(f"\n❌ Embedding type '{embedding_type}' not found in {step}/{batch}")
            print(f"   Available: {list(self.embeddings[step][batch].keys())}")
            return
        
        stats = self.get_stats(step, batch, embedding_type)
        
        print(f"\n📊 Statistics for {step}/{batch}/{embedding_type}:")
        print(f"  Shape:   {stats['shape']}")
        print(f"  Mean:    {stats['mean']:.6f}")
        print(f"  Std:     {stats['std']:.6f}")
        print(f"  Min:     {stats['min']:.6f}")
        print(f"  Max:     {stats['max']:.6f}")
        print(f"  Median:  {stats['median']:.6f}")
        print(f"  Q25-Q75: {stats['q25']:.6f} - {stats['q75']:.6f}\n")
    
    def draw_gaussian_1d(self, step: Optional[str] = None, batch: Optional[str] = None,
                         embedding_type: str = "gt", dim: int = 0, save_path: Optional[str] = None) -> None:
        """
        Draw 1D Gaussian distribution for a specific dimension.
        
        Args:
            step: Validation step
            batch: Batch folder
            embedding_type: Type of embedding
            dim: Dimension index (will be flattened)
            save_path: Path to save figure
        """
        if step is None:
            step = list(self.embeddings.keys())[0]
        if batch is None:
            batch = list(self.embeddings[step].keys())[0]
        
        embedding = self.embeddings[step][batch][embedding_type]
        flattened = embedding.reshape(-1)
        
        if dim >= len(flattened):
            print(f"⚠️  Dimension {dim} out of range (max: {len(flattened)-1})")
            return
        
        data_point = flattened[dim]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Histogram with Gaussian overlay
        stats_dict = self.get_stats(step, batch, embedding_type)
        mean = stats_dict['mean']
        std = stats_dict['std']
        
        ax1.hist(flattened, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        x = np.linspace(mean - 4*std, mean + 4*std, 100)
        ax1.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2, label='Gaussian N(μ, σ²)')
        ax1.axvline(data_point, color='green', linestyle='--', linewidth=2, label=f'Dim {dim} value')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Flattened Distribution ({embedding_type})\nμ={mean:.4f}, σ={std:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot
        stats.probplot(flattened, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Check)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"gaussian_1d_{embedding_type}_dim{dim}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved 1D Gaussian plot to: {save_path}")
        plt.close()
    
    def draw_gaussian_2d(self, step: Optional[str] = None, batch: Optional[str] = None,
                         embedding_type: str = "gt", dims: Tuple[int, int] = (0, 1),
                         save_path: Optional[str] = None) -> None:
        """
        Draw 2D Gaussian distribution.
        
        Args:
            step: Validation step
            batch: Batch folder
            embedding_type: Type of embedding
            dims: Tuple of (dim1, dim2) to plot
            save_path: Path to save figure
        """
        if step is None:
            step = list(self.embeddings.keys())[0]
        if batch is None:
            batch = list(self.embeddings[step].keys())[0]
        
        embedding = self.embeddings[step][batch][embedding_type]
        flattened = embedding.reshape(-1)
        
        if max(dims) >= len(flattened):
            print(f"⚠️  Dimensions {dims} out of range (max: {len(flattened)-1})")
            return
        
        x = flattened[dims[0]]
        y = flattened[dims[1]]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create 2D density plot
        data_2d = np.column_stack([
            embedding.reshape(embedding.shape[0], -1)[:, dims[0]],
            embedding.reshape(embedding.shape[0], -1)[:, dims[1]]
        ])
        
        # Compute mean and covariance
        mean = data_2d.mean(axis=0)
        cov = np.cov(data_2d.T)
        
        # Plot data points
        ax.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5, s=20, color='blue', label='Data')
        
        # Plot Gaussian contours
        from scipy.stats import multivariate_normal
        x_range = np.linspace(data_2d[:, 0].min() - 2, data_2d[:, 0].max() + 2, 100)
        y_range = np.linspace(data_2d[:, 1].min() - 2, data_2d[:, 1].max() + 2, 100)
        X, Y = np.meshgrid(x_range, y_range)
        pos = np.dstack((X, Y))
        
        try:
            rv = multivariate_normal(mean, cov)
            Z = rv.pdf(pos)
            ax.contour(X, Y, Z, levels=5, colors='red', alpha=0.6, linewidths=1.5)
            ax.contourf(X, Y, Z, levels=5, colors='red', alpha=0.1)
        except Exception as e:
            print(f"⚠️  Could not plot contours: {e}")
        
        ax.scatter(*mean, color='red', s=100, marker='x', linewidth=3, label='Mean')
        ax.set_xlabel(f'Dimension {dims[0]}')
        ax.set_ylabel(f'Dimension {dims[1]}')
        ax.set_title(f'2D Gaussian Distribution ({embedding_type})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = f"gaussian_2d_{embedding_type}_dims{dims[0]}{dims[1]}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved 2D Gaussian plot to: {save_path}")
        plt.close()
    
    def draw_violin_plot(self, step: Optional[str] = None, batch: Optional[str] = None,
                         embedding_type: str = "gt", n_dims: int = 64,
                         save_path: Optional[str] = None) -> None:
        """
        Draw violin plots for first N dimensions.
        
        Args:
            step: Validation step
            batch: Batch folder
            embedding_type: Type of embedding
            n_dims: Number of dimensions to show
            save_path: Path to save figure
        """
        if step is None:
            step = list(self.embeddings.keys())[0]
        if batch is None:
            batch = list(self.embeddings[step].keys())[0]
        
        embedding = self.embeddings[step][batch][embedding_type]
        flattened = embedding.reshape(embedding.shape[0], -1)  # (batch, dims)
        
        n_dims = min(n_dims, flattened.shape[1])
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Prepare data for violin plot
        data_list = [flattened[:, i] for i in range(n_dims)]
        
        sns.violinplot(data=data_list, ax=ax)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.set_title(f'Violin Plot - First {n_dims} Dimensions ({embedding_type})')
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path is None:
            save_path = f"violin_plot_{embedding_type}_{n_dims}dims.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved violin plot to: {save_path}")
        plt.close()
    
    def draw_comparison(self, step: Optional[str] = None, batch: Optional[str] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Compare different embedding types (gt vs prediction, etc).
        
        Args:
            step: Validation step
            batch: Batch folder
            save_path: Path to save figure
        """
        if step is None:
            step = list(self.embeddings.keys())[0]
        if batch is None:
            batch = list(self.embeddings[step].keys())[0]
        
        fig, axes = plt.subplots(1, len(self.embeddings[step][batch]), figsize=(15, 5))
        if len(self.embeddings[step][batch]) == 1:
            axes = [axes]
        
        for ax, (emb_type, embedding) in zip(axes, self.embeddings[step][batch].items()):
            flattened = embedding.reshape(-1)
            mean = np.mean(flattened)
            std = np.std(flattened)
            
            ax.hist(flattened, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
            x = np.linspace(mean - 4*std, mean + 4*std, 100)
            ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2)
            ax.set_title(f'{emb_type}\nμ={mean:.4f}, σ={std:.4f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"comparison_{step}_{batch}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved comparison plot to: {save_path}")
        plt.close()
    
    def list_available(self) -> None:
        """List all available embeddings."""
        if not self.embeddings:
            print("\n❌ No embeddings loaded!")
            return
        
        print("\n📋 Available Embeddings:\n")
        for step in self.embeddings:
            print(f"  {step}/")
            if not self.embeddings[step]:
                print("    (empty)")
            for batch in self.embeddings[step]:
                types = list(self.embeddings[step][batch].keys())
                if not types:
                    print(f"    {batch}: (no data)")
                else:
                    print(f"    {batch}: {', '.join(types)}")


def main():
    parser = argparse.ArgumentParser(description='Analyze saved embeddings')
    parser.add_argument('embeddings_dir', type=str, help='Path to embeddings directory')
    parser.add_argument('--analyze-1d', action='store_true', help='Draw 1D Gaussian for first dimension')
    parser.add_argument('--analyze-2d', action='store_true', help='Draw 2D Gaussian for first two dimensions')
    parser.add_argument('--analyze-violin', action='store_true', help='Draw violin plots')
    parser.add_argument('--compare', action='store_true', help='Compare embedding types')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--step', type=str, default=None, help='Specific step to analyze')
    parser.add_argument('--batch', type=str, default=None, help='Specific batch to analyze')
    parser.add_argument('--type', type=str, default='gt', help='Embedding type to analyze')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    analyzer = EmbeddingAnalyzer(args.embeddings_dir)
    
    # List available embeddings
    analyzer.list_available()
    
    # Check if any embeddings were loaded
    if not analyzer.embeddings or all(not v for v in analyzer.embeddings.values()):
        print("\n❌ ERROR: No embeddings data found!")
        print(f"\nExpected structure:")
        print(f"  {args.embeddings_dir}/validation_step_*/batch_*/gt.npz")
        return
    
    # Print stats
    analyzer.print_stats(args.step, args.batch, args.type)
    
    # Run analyses (only if embeddings exist)
    if analyzer.embeddings and any(v for v in analyzer.embeddings.values()):
        if args.analyze_1d or args.all:
            save_path = Path(args.output_dir) / f"gaussian_1d_{args.type}.png"
            analyzer.draw_gaussian_1d(args.step, args.batch, args.type, save_path=str(save_path))
        
        if args.analyze_2d or args.all:
            save_path = Path(args.output_dir) / f"gaussian_2d_{args.type}.png"
            analyzer.draw_gaussian_2d(args.step, args.batch, args.type, save_path=str(save_path))
        
        if args.analyze_violin or args.all:
            save_path = Path(args.output_dir) / f"violin_{args.type}.png"
            analyzer.draw_violin_plot(args.step, args.batch, args.type, save_path=str(save_path))
        
        if args.compare or args.all:
            save_path = Path(args.output_dir) / f"comparison_{args.step}.png"
            analyzer.draw_comparison(args.step, args.batch, save_path=str(save_path))


if __name__ == '__main__':
    main()
