from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import google.generativeai as genai
from dotenv import load_dotenv

from pathlib import Path

# Load environment variables from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ColumnStats(BaseModel):
    mean: Optional[str] = None
    median: Optional[str] = None
    std: Optional[str] = None
    min: Optional[str] = None
    max: Optional[str] = None

class ColumnProfile(BaseModel):
    name: str
    type: str
    nullPercent: str
    uniqueCount: int
    stats: Optional[ColumnStats] = None
    sampleValues: Optional[List[str]] = None

class Correlation(BaseModel):
    col1: str
    col2: str
    correlation: str

class AnalysisData(BaseModel):
    numRows: int
    numCols: int
    totalNullPercent: str
    columnAnalysis: List[ColumnProfile]
    correlations: List[Correlation]

@app.post("/summarize")
async def generate_summary(data: AnalysisData):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Reconstruct the prompt on the server side
    prompt = f"""You are a data analyst. Provide a concise summary (200-300 words) of this dataset analysis.

Dataset Overview:
- Rows: {data.numRows}
- Columns: {data.numCols}
- Overall missing data: {data.totalNullPercent}%

Column Details:
"""

    for col in data.columnAnalysis:
        prompt += f"- {col.name} ({col.type}): {col.nullPercent}% missing, {col.uniqueCount} unique values\n"
        if col.type == 'numeric' and col.stats:
            prompt += f"  Stats: Mean={col.stats.mean}, Std={col.stats.std}, Range=[{col.stats.min}, {col.stats.max}]\n"
        elif col.sampleValues:
            prompt += f"  Sample values: {', '.join(col.sampleValues)}\n"

    if data.correlations:
        prompt += "\nNotable Correlations:\n"
        for c in data.correlations:
            prompt += f"- {c.col1} ↔ {c.col2}: {c.correlation}\n"

    prompt += "\nProvide insights about data quality, patterns, and potential areas of interest for analysis."

    try:
        response = model.generate_content(prompt)
        return {"summary": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
