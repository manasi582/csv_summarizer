
import streamlit as st
import os
import tempfile
from backend.main import DataAnalyzer, SummaryGenerator, Visualizer

# Page Config
st.set_page_config(
    page_title="CSV Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Professional" look
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    with st.sidebar:
        st.title("📊 CSV Analyzer")
        st.markdown("---")
        st.markdown("Upload a CSV file to generate automated insights, visualizations, and an AI-powered summary.")
        st.info("💡 Pro Tip: Ensure your CSV has clear headers.")

    # Main Content
    st.title("CSV Analyzer")
    st.markdown("### Analyze. Visualize. Optimize.")

    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            # Create a temporary file to pass to DataAnalyzer
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Initialize Backend Classes
            analyzer = DataAnalyzer(tmp_path)
            visualizer = Visualizer(output_dir="streamlit_output")
            summary_gen = SummaryGenerator()  # Picks up GEMINI_API_KEY from env automatically

            # 1. Load & Profile Data
            with st.spinner("Profiling dataset..."):
                analyzer.load_data()
                profile = analyzer.profile_dataset()

            # 2. Key Metrics Row
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{profile.num_rows:,}")
            with col2:
                st.metric("Columns", f"{profile.num_cols}")
            with col3:
                st.metric("Missing Data", f"{profile.total_null_percent}%")
            with col4:
                st.metric("Duplicates (Rows)", f"{analyzer.df.duplicated().sum():,}")

            # 3. AI Summary
            st.markdown("### 🤖 AI Executive Summary")
            with st.spinner("Generating insights with Gemini..."):
                summary_text = summary_gen.generate_summary(profile)
                st.success(summary_text)

            # 4. Visualizations
            st.markdown("### 📈 Visualizations")
            
            # create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Missing Values"])

            with tab1:
                st.subheader("Numeric Distributions")
                dist_path = visualizer.plot_distributions(analyzer.df, filename="dist_temp.png")
                if dist_path and os.path.exists(dist_path):
                    st.image(dist_path, use_column_width=True)
                else:
                    st.info("No numeric columns to plot.")

            with tab2:
                st.subheader("Correlation Matrix")
                corr_path = visualizer.plot_correlation_matrix(analyzer.df, filename="corr_temp.png")
                if corr_path and os.path.exists(corr_path):
                    st.image(corr_path, use_column_width=True)
                else:
                    st.info("Not enough numeric columns for correlations.")

            with tab3:
                st.subheader("Missing Value Heatmap")
                miss_path = visualizer.plot_missing_values(analyzer.df, filename="miss_temp.png")
                if miss_path and os.path.exists(miss_path):
                    st.image(miss_path, use_column_width=True)
                else:
                    st.success("No missing values detected!")

            # 5. Data Preview
            st.markdown("### 📄 Data Preview")
            st.dataframe(analyzer.df.head(10))

            # Cleanup
            os.unlink(tmp_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

if __name__ == "__main__":
    main()
