
"""
CSV Analyzer - Analyze. Visualize. Optimize.
Author: Senior Data Engineer
Version: 1.0.0
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    """Core data analysis engine for CSV profiling."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self.df: Optional[pd.DataFrame] = None
        self.profile: Optional[DatasetProfile] = None
        logger.info(f"Initialized analyzer for: {self.filepath.name}")

    def load_data(self) -> None:
        """Load CSV data using Pandas."""
        try:
            # Simplified loading: let Pandas handle encoding detection best effort
            self.df = pd.read_csv(self.filepath, encoding_errors='replace')
            logger.info(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns.")
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

    def analyze_column(self, col_name: str) -> ColumnProfile:
        """Perform detailed analysis on a single column."""
        series = self.df[col_name]
        null_count = series.isna().sum()
        unique_count = series.nunique()
        
        # Determine if numeric
        is_numeric = pd.api.types.is_numeric_dtype(series)
        
        profile = ColumnProfile(
            name=col_name,
            dtype='numeric' if is_numeric else 'categorical',
            null_count=null_count,
            null_percent=round((null_count / len(series)) * 100, 2),
            unique_count=unique_count,
            cardinality=round((unique_count / len(series)) * 100, 2)
        )
        
        if is_numeric:
            profile.mean = round(series.mean(), 2) if not series.isna().all() else None
            profile.median = round(series.median(), 2) if not series.isna().all() else None
            profile.std = round(series.std(), 2) if not series.isna().all() else None
            profile.min_val = round(series.min(), 2) if not series.isna().all() else None
            profile.max_val = round(series.max(), 2) if not series.isna().all() else None
        else:
            profile.sample_values = series.dropna().unique()[:5].tolist()
        
        return profile
    
    def detect_correlations(self, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """Detect correlations between numeric columns."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2: return []
        
        corr_matrix = self.df[numeric_cols].corr()
        correlations = []
        
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold and not np.isnan(corr_val):
                    correlations.append((numeric_cols[i], numeric_cols[j], round(corr_val, 3)))
        return correlations
    
    def profile_dataset(self) -> DatasetProfile:
        """Generate comprehensive dataset profile."""
        if self.df is None: raise ValueError("Data not loaded.")
        
        column_profiles = [self.analyze_column(col) for col in self.df.columns]
        total_cells = len(self.df) * len(self.df.columns)
        total_null_percent = round((self.df.isna().sum().sum() / total_cells) * 100, 2)
        
        self.profile = DatasetProfile(
            filename=self.filepath.name,
            num_rows=len(self.df),
            num_cols=len(self.df.columns),
            total_null_percent=total_null_percent,
            column_profiles=column_profiles,
            correlations=self.detect_correlations(),
            memory_usage_mb=round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        )
        return self.profile


class Visualizer:
    """Generates visualizations for data analysis."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_missing_values(self, df: pd.DataFrame, filename: str = "missing_values.png") -> str:
        missing = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
        if missing.sum() == 0: return ""
        
        plt.figure()
        missing.plot(kind='bar', color=['#ef4444' if x > 10 else '#06b6d4' for x in missing])
        plt.title('Missing Values (%)'); plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path); plt.close()
        return str(output_path)
    
    def plot_correlation_matrix(self, df: pd.DataFrame, filename: str = "correlation_matrix.png") -> str:
        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) < 2: return ""
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix'); plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path); plt.close()
        return str(output_path)
    
    def plot_distributions(self, df: pd.DataFrame, filename: str = "distributions.png") -> str:
        numeric = df.select_dtypes(include=[np.number]).columns
        if len(numeric) == 0: return ""
        
        df[numeric].hist(bins=30, figsize=(15, 10), color='#06b6d4', edgecolor='black')
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path); plt.close()
        return str(output_path)


class SummaryGenerator:
    """Generates natural language summaries using LLM."""
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
    
    def generate_summary(self, profile: DatasetProfile) -> str:
        if not self.model: return "API Key not found. Please check .env file."
        
        prompt = f"""
        Analyze this dataset:
        - Rows: {profile.num_rows}
        - Columns: {profile.num_cols}
        - Missing Data: {profile.total_null_percent}%
        
        Columns:
        {chr(10).join([f"- {c.name} ({c.dtype})" for c in profile.column_profiles])}
        
        Correlations:
        {chr(10).join([f"- {c[0]} vs {c[1]}: {c[2]}" for c in profile.correlations])}
        
        Write a professional 100-word executive summary.
        """
        try:
            return self.model.generate_content(prompt).text
        except Exception as e:
            return f"Error generating summary: {str(e)}"
