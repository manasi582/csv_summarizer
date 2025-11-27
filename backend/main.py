
"""
CSV Analyzer - Professional Data Profiling Suite
Author: Senior Data Engineer
Version: 1.0.0
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from anthropic import Anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """Data class for column statistics."""
    name: str
    dtype: str
    null_count: int
    null_percent: float
    unique_count: int
    cardinality: float
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    sample_values: Optional[List[str]] = None


@dataclass
class DatasetProfile:
    """Data class for dataset-level statistics."""
    filename: str
    num_rows: int
    num_cols: int
    total_null_percent: float
    column_profiles: List[ColumnProfile]
    correlations: List[Tuple[str, str, float]]
    memory_usage_mb: float


class DataAnalyzer:
    """
    Core data analysis engine for CSV profiling.
    
    Performs comprehensive statistical analysis including:
    - Column-level profiling
    - Missing value analysis
    - Correlation detection
    - Data type inference
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the analyzer with a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a CSV
        """
        self.filepath = Path(filepath)
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if self.filepath.suffix.lower() != '.csv':
            raise ValueError("Only CSV files are supported")
        
        self.df: Optional[pd.DataFrame] = None
        self.profile: Optional[DatasetProfile] = None
        
        logger.info(f"Initialized analyzer for: {self.filepath.name}")
    
class DataAnalyzer:
    """
    Core data analysis engine for CSV profiling.
    
    Performs comprehensive statistical analysis including:
    - Column-level profiling
    - Missing value analysis
    - Correlation detection
    - Data type inference
    """

    def __init__(self, filepath: str):
        """
        Initialize the analyzer with a CSV file.
        """
        from pathlib import Path

        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        if self.filepath.suffix.lower() != '.csv':
            raise ValueError("Only CSV files are supported")

        self.df = None
        self.profile = None
        logger.info(f"Initialized analyzer for: {self.filepath.name}")

    def load_data(self) -> None:
        """
        Load CSV data with robust path + encoding handling and clear logs.
        """
        try:
            abs_path = self.filepath.resolve()
            logger.info(f"Resolved CSV path: {abs_path}")
            if not abs_path.exists():
                raise FileNotFoundError(f"File not found at: {abs_path}")

            # Try to detect encoding if chardet is available
            detected_enc = None
            try:
                import chardet  # pip install chardet
                with open(abs_path, "rb") as f:
                    chunk = f.read(20000)
                detected = chardet.detect(chunk)
                detected_enc = detected.get("encoding") or None
                logger.info(f"chardet detected encoding: {detected_enc} (confidence={detected.get('confidence')})")
            except Exception:
                logger.info("chardet not available or failed; will try common encodings.")

            # Ordered list of encodings to try
            candidates = ["utf-8", "utf-8-sig"]
            if detected_enc and detected_enc.lower() not in [c.lower() for c in candidates]:
                candidates.append(detected_enc)
            candidates += ["cp1252", "ISO-8859-1", "latin1"]

            last_err = None
            for enc in candidates:
                for use_python_engine in (False, True):
                    try:
                        logger.info(f"Trying read_csv with encoding='{enc}'"
                                    f"{' and engine=python' if use_python_engine else ''}")
                        self.df = pd.read_csv(
                            abs_path,
                            encoding=enc,
                            engine=("python" if use_python_engine else None)
                        )
                        logger.info(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns using {enc}"
                                    f"{' + python engine' if use_python_engine else ''}")
                        if self.df.empty:
                            raise pd.errors.EmptyDataError("CSV file is empty")
                        return  # ✅ success
                    except Exception as e:
                        last_err = e
                        logger.warning(f"Failed with encoding='{enc}'"
                                       f"{' + python engine' if use_python_engine else ''}: {e}")

            # If we reach here, all attempts failed
            raise last_err if last_err else RuntimeError("Failed to load CSV with all fallbacks.")

        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse CSV: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading CSV: {e}")
            raise


    
    def analyze_column(self, col_name: str) -> ColumnProfile:
        """
        Perform detailed analysis on a single column.
        
        Args:
            col_name: Name of the column to analyze
            
        Returns:
            ColumnProfile object with statistics
        """
        series = self.df[col_name]
        null_count = series.isna().sum()
        null_percent = (null_count / len(series)) * 100
        unique_count = series.nunique()
        cardinality = (unique_count / len(series)) * 100
        
        # Determine if numeric
        is_numeric = pd.api.types.is_numeric_dtype(series)
        
        profile = ColumnProfile(
            name=col_name,
            dtype='numeric' if is_numeric else 'categorical',
            null_count=null_count,
            null_percent=round(null_percent, 2),
            unique_count=unique_count,
            cardinality=round(cardinality, 2)
        )
        
        if is_numeric:
            # Calculate numeric statistics
            profile.mean = round(series.mean(), 2) if not series.isna().all() else None
            profile.median = round(series.median(), 2) if not series.isna().all() else None
            profile.std = round(series.std(), 2) if not series.isna().all() else None
            profile.min_val = round(series.min(), 2) if not series.isna().all() else None
            profile.max_val = round(series.max(), 2) if not series.isna().all() else None
        else:
            # Get sample values for categorical
            profile.sample_values = series.dropna().unique()[:5].tolist()
        
        return profile
    
    def detect_correlations(self, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """
        Detect correlations between numeric columns.
        
        Args:
            threshold: Minimum correlation coefficient to report
            
        Returns:
            List of tuples (col1, col2, correlation)
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.info("Insufficient numeric columns for correlation analysis")
            return []
        
        corr_matrix = self.df[numeric_cols].corr()
        correlations = []
        
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) >= threshold and not np.isnan(corr_val):
                    correlations.append((
                        numeric_cols[i],
                        numeric_cols[j],
                        round(corr_val, 3)
                    ))
        
        logger.info(f"Detected {len(correlations)} significant correlations")
        return correlations
    
    def profile_dataset(self) -> DatasetProfile:
        """
        Generate comprehensive dataset profile.
        
        Returns:
            DatasetProfile object with all statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Starting dataset profiling...")
        
        # Analyze each column
        column_profiles = [self.analyze_column(col) for col in self.df.columns]
        
        # Detect correlations
        correlations = self.detect_correlations()
        
        # Calculate overall statistics
        total_cells = len(self.df) * len(self.df.columns)
        total_nulls = self.df.isna().sum().sum()
        total_null_percent = round((total_nulls / total_cells) * 100, 2)
        
        # Memory usage
        memory_usage_mb = round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        
        self.profile = DatasetProfile(
            filename=self.filepath.name,
            num_rows=len(self.df),
            num_cols=len(self.df.columns),
            total_null_percent=total_null_percent,
            column_profiles=column_profiles,
            correlations=correlations,
            memory_usage_mb=memory_usage_mb
        )
        
        logger.info("Dataset profiling complete")
        return self.profile


class Visualizer:
    """
    Generates visualizations for data analysis.
    
    Creates professional plots including:
    - Missing value heatmaps
    - Distribution histograms
    - Correlation matrices
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_missing_values(self, df: pd.DataFrame, filename: str = "missing_values.png") -> str:
        """
        Create bar plot of missing values per column.
        
        Args:
            df: DataFrame to analyze
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        missing_percent = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
        
        if missing_percent.sum() == 0:
            logger.info("No missing values to plot")
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#ef4444' if x > 10 else '#06b6d4' for x in missing_percent]
        missing_percent.plot(kind='bar', ax=ax, color=colors)
        
        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Missing (%)', fontsize=12)
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.3, label='10% threshold')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved missing values plot: {output_path}")
        return str(output_path)
    
    def plot_correlation_matrix(self, df: pd.DataFrame, filename: str = "correlation_matrix.png") -> str:
        """
        Create heatmap of correlation matrix.
        
        Args:
            df: DataFrame to analyze
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.info("Insufficient numeric columns for correlation matrix")
            return ""
        
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation matrix: {output_path}")
        return str(output_path)
    
    def plot_distributions(self, df: pd.DataFrame, filename: str = "distributions.png") -> str:
        """
        Create histograms for numeric columns.
        
        Args:
            df: DataFrame to analyze
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.info("No numeric columns to plot")
            return ""
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            df[col].hist(bins=30, ax=axes[idx], color='#06b6d4', edgecolor='black')
            axes[idx].set_title(f'{col} Distribution', fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved distributions plot: {output_path}")
        return str(output_path)


class SummaryGenerator:
    """
    Generates natural language summaries using LLM.
    
    Uses Anthropic's Claude API to create human-readable
    insights from statistical analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize summary generator.
        
        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
        """
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            logger.warning("No API key provided. Summary generation will be skipped.")
            self.client = None
        else:
            self.client = Anthropic(api_key=api_key)
    
    def generate_summary(self, profile: DatasetProfile) -> str:
        """
        Generate natural language summary from profile.
        
        Args:
            profile: Dataset profile object
            
        Returns:
            Natural language summary string
        """
        if self.client is None:
            return self._generate_basic_summary(profile)
        
        prompt = self._build_prompt(profile)
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = message.content[0].text
            logger.info("Generated AI summary successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate AI summary: {e}")
            return self._generate_basic_summary(profile)
    
    def _build_prompt(self, profile: DatasetProfile) -> str:
        """Build prompt for LLM."""
        column_details = "\n".join([
            f"- {col.name} ({col.dtype}): {col.null_percent}% missing, "
            f"{col.unique_count} unique values"
            + (f", Mean={col.mean}, Std={col.std}" if col.dtype == 'numeric' else "")
            for col in profile.column_profiles
        ])
        
        correlations = "\n".join([
            f"- {c[0]} ↔ {c[1]}: {c[2]}"
            for c in profile.correlations
        ]) if profile.correlations else "None detected"
        
        return f"""You are a data analyst. Provide a concise summary (200-300 words) of this dataset analysis.

Dataset Overview:
- Rows: {profile.num_rows:,}
- Columns: {profile.num_cols}
- Overall missing data: {profile.total_null_percent}%
- Memory usage: {profile.memory_usage_mb} MB

Column Details:
{column_details}

Notable Correlations:
{correlations}

Provide insights about data quality, patterns, and potential areas of interest for analysis."""
    
    def _generate_basic_summary(self, profile: DatasetProfile) -> str:
        """Generate basic summary without LLM."""
        numeric_cols = sum(1 for c in profile.column_profiles if c.dtype == 'numeric')
        categorical_cols = profile.num_cols - numeric_cols
        
        summary = f"""Dataset Analysis Summary

The dataset '{profile.filename}' contains {profile.num_rows:,} rows and {profile.num_cols} columns.

Data Composition:
- {numeric_cols} numeric columns
- {categorical_cols} categorical columns
- {profile.memory_usage_mb} MB memory usage

Data Quality:
- Overall completeness: {100 - profile.total_null_percent:.1f}%
- Missing data: {profile.total_null_percent}%
- Columns with >10% missing: {sum(1 for c in profile.column_profiles if c.null_percent > 10)}

{'Correlations: ' + str(len(profile.correlations)) + ' significant correlations detected.' if profile.correlations else 'No significant correlations detected.'}

This dataset appears to be {'well-structured with minimal missing data' if profile.total_null_percent < 5 else 'requires data cleaning due to missing values'}."""
        
        return summary


class ReportGenerator:
    """
    Generates comprehensive analysis reports.
    
    Creates formatted text reports with all statistics,
    visualizations, and summaries.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(
        self,
        profile: DatasetProfile,
        summary: str,
        visualizations: Dict[str, str]
    ) -> str:
        """
        Generate comprehensive text report.
        
        Args:
            profile: Dataset profile
            summary: AI-generated summary
            visualizations: Dict of visualization paths
            
        Returns:
            Path to saved report
        """
        report_lines = [
            "CSV ANALYSIS REPORT",
            "=" * 80,
            "",
            "FILE INFORMATION",
            "-" * 80,
            f"Filename: {profile.filename}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analyst: CSV Analyzer v1.0",
            "",
            "=" * 80,
            "EXECUTIVE SUMMARY",
            "=" * 80,
            "",
            summary,
            "",
            "=" * 80,
            "DATASET METRICS",
            "=" * 80,
            "",
            f"Dimensions: {profile.num_rows:,} rows × {profile.num_cols} columns",
            f"Memory Usage: {profile.memory_usage_mb} MB",
            f"Data Completeness: {100 - profile.total_null_percent:.2f}%",
            f"Missing Values: {profile.total_null_percent}%",
            "",
            "=" * 80,
            "COLUMN-LEVEL ANALYSIS",
            "=" * 80,
            ""
        ]
        
        # Add column details
        for idx, col in enumerate(profile.column_profiles, 1):
            report_lines.extend([
                f"[{idx}] {col.name}",
                "-" * 40,
                f"Type: {col.dtype.upper()}",
                f"Missing: {col.null_percent}% ({col.null_count} values)",
                f"Unique Values: {col.unique_count}",
                f"Cardinality: {col.cardinality}%"
            ])
            
            if col.dtype == 'numeric':
                report_lines.extend([
                    "",
                    "Statistical Summary:",
                    f"  • Mean: {col.mean}",
                    f"  • Median: {col.median}",
                    f"  • Std Dev: {col.std}",
                    f"  • Range: [{col.min_val}, {col.max_val}]",
                    f"  • Coefficient of Variation: {(col.std / col.mean * 100):.2f}%" if col.mean and col.mean != 0 else ""
                ])
            else:
                report_lines.extend([
                    "",
                    "Sample Values:",
                    *[f"  • {val}" for val in (col.sample_values or [])[:5]]
                ])
            
            report_lines.append("")
        
        # Add correlations
        if profile.correlations:
            report_lines.extend([
                "=" * 80,
                "CORRELATION ANALYSIS",
                "=" * 80,
                "",
                f"Detected {len(profile.correlations)} significant correlation(s) (|r| > 0.3):",
                ""
            ])
            
            for idx, (col1, col2, corr) in enumerate(profile.correlations, 1):
                strength = 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.5 else 'Weak'
                direction = 'Positive' if corr > 0 else 'Negative'
                
                report_lines.extend([
                    f"[{idx}] {col1} ↔ {col2}",
                    f"    Pearson's r: {corr}",
                    f"    Strength: {strength}",
                    f"    Direction: {direction}",
                    ""
                ])
        
        # Add data quality assessment
        complete_cols = sum(1 for c in profile.column_profiles if c.null_percent == 0)
        high_missing_cols = sum(1 for c in profile.column_profiles if c.null_percent > 10)
        high_cardinality_cols = sum(1 for c in profile.column_profiles if c.cardinality > 90)
        
        report_lines.extend([
            "=" * 80,
            "DATA QUALITY ASSESSMENT",
            "=" * 80,
            "",
            f"Completeness Score: {100 - profile.total_null_percent:.1f}%",
            f"Complete Columns: {complete_cols}/{profile.num_cols}",
            f"Columns with >10% Missing: {high_missing_cols}",
            f"High Cardinality Columns: {high_cardinality_cols}",
            ""
        ])
        
        # Add visualization references
        if visualizations:
            report_lines.extend([
                "=" * 80,
                "VISUALIZATIONS",
                "=" * 80,
                "",
                "Generated plots:",
                *[f"  • {name}: {path}" for name, path in visualizations.items()],
                ""
            ])
        
        report_lines.extend([
            "=" * 80,
            "End of Report",
            ""
        ])
        
        # Write report
        report_text = "\n".join(report_lines)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"{profile.filename.replace('.csv', '')}_{timestamp}_analysis.txt"
        
        report_path.write_text(report_text)
        logger.info(f"Report saved: {report_path}")
        
        return str(report_path)


def main(csv_file: str, api_key: Optional[str] = None):
    """
    Main execution function.
    
    Args:
        csv_file: Path to CSV file
        api_key: Optional Anthropic API key
    """
    print("=" * 80)
    print("CSV ANALYZER")
    print("=" * 80)
    print()
    
    try:
        # Initialize components
        analyzer = DataAnalyzer(csv_file)
        visualizer = Visualizer()
        summary_gen = SummaryGenerator(api_key)
        report_gen = ReportGenerator()
        
        # Load and analyze data
        print(f"Loading: {csv_file}")
        analyzer.load_data()
        
        print("Profiling dataset...")
        profile = analyzer.profile_dataset()
        
        # Generate visualizations
        print("Creating visualizations...")
        visualizations = {}
        
        if plot_path := visualizer.plot_missing_values(analyzer.df):
            visualizations['Missing Values'] = plot_path
        
        if plot_path := visualizer.plot_correlation_matrix(analyzer.df):
            visualizations['Correlation Matrix'] = plot_path
        
        if plot_path := visualizer.plot_distributions(analyzer.df):
            visualizations['Distributions'] = plot_path
        
        # Generate summary
        print("Generating AI summary...")
        summary = summary_gen.generate_summary(profile)
        
        # Generate report
        print("Creating report...")
        report_path = report_gen.generate_report(profile, summary, visualizations)
        
        # Print summary
        print()
        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print()
        print(f"Dataset: {profile.filename}")
        print(f"Rows: {profile.num_rows:,}")
        print(f"Columns: {profile.num_cols}")
        print(f"Missing Data: {profile.total_null_percent}%")
        print(f"Correlations: {len(profile.correlations)}")
        print()
        print(f"Report saved: {report_path}")
        print(f"Visualizations: {len(visualizations)}")
        print()
        print("SUMMARY:")
        print("-" * 80)
        print(summary)
        print()
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CSV Analyzer - Professional Data Profiling")
    parser.add_argument("csv_file", help="Path to CSV file")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    
    args = parser.parse_args()
    main(args.csv_file, args.api_key)

