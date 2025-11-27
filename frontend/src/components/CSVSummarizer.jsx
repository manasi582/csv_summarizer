import React, { useState } from 'react';
import { Upload, FileText, BarChart3, AlertCircle, Download, Loader2, TrendingUp, Database, Activity, CheckCircle, XCircle } from 'lucide-react';
import Papa from 'papaparse';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, PieChart, Pie } from 'recharts';

const CSVSummarizer = () => {
    const [file, setFile] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [analysis, setAnalysis] = useState(null);
    const [summary, setSummary] = useState('');
    const [error, setError] = useState('');

    const analyzeCSV = (data, headers) => {
        const numRows = data.length;
        const numCols = headers.length;

        const columnAnalysis = headers.map(col => {
            const values = data.map(row => row[col]).filter(v => v !== null && v !== undefined && v !== '');
            const nullCount = numRows - values.length;
            const uniqueValues = new Set(values);

            const numericValues = values.map(v => parseFloat(v)).filter(v => !isNaN(v));
            const isNumeric = numericValues.length > values.length * 0.8;

            let stats = {};
            if (isNumeric && numericValues.length > 0) {
                const sorted = [...numericValues].sort((a, b) => a - b);
                const sum = numericValues.reduce((a, b) => a + b, 0);
                const mean = sum / numericValues.length;
                const variance = numericValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / numericValues.length;

                stats = {
                    mean: mean.toFixed(2),
                    median: sorted[Math.floor(sorted.length / 2)].toFixed(2),
                    std: Math.sqrt(variance).toFixed(2),
                    min: Math.min(...numericValues).toFixed(2),
                    max: Math.max(...numericValues).toFixed(2)
                };
            }

            return {
                name: col,
                type: isNumeric ? 'numeric' : 'categorical',
                nullCount,
                nullPercent: ((nullCount / numRows) * 100).toFixed(1),
                uniqueCount: uniqueValues.size,
                stats,
                sampleValues: Array.from(uniqueValues).slice(0, 5)
            };
        });

        const numericCols = columnAnalysis.filter(c => c.type === 'numeric');
        const correlations = [];

        for (let i = 0; i < numericCols.length; i++) {
            for (let j = i + 1; j < numericCols.length; j++) {
                const col1 = numericCols[i].name;
                const col2 = numericCols[j].name;

                const pairs = data.map(row => ({
                    x: parseFloat(row[col1]),
                    y: parseFloat(row[col2])
                })).filter(p => !isNaN(p.x) && !isNaN(p.y));

                if (pairs.length > 0) {
                    const meanX = pairs.reduce((a, b) => a + b.x, 0) / pairs.length;
                    const meanY = pairs.reduce((a, b) => a + b.y, 0) / pairs.length;

                    const cov = pairs.reduce((a, b) => a + (b.x - meanX) * (b.y - meanY), 0) / pairs.length;
                    const stdX = Math.sqrt(pairs.reduce((a, b) => a + Math.pow(b.x - meanX, 2), 0) / pairs.length);
                    const stdY = Math.sqrt(pairs.reduce((a, b) => a + Math.pow(b.y - meanY, 2), 0) / pairs.length);

                    const corr = cov / (stdX * stdY);
                    if (Math.abs(corr) > 0.3) {
                        correlations.push({ col1, col2, correlation: corr.toFixed(3) });
                    }
                }
            }
        }

        return {
            numRows,
            numCols,
            columnAnalysis,
            correlations,
            totalNullPercent: ((columnAnalysis.reduce((a, b) => a + b.nullCount, 0) / (numRows * numCols)) * 100).toFixed(2)
        };
    };

    const generateLLMSummary = async (analysisData) => {
        const prompt = `You are a data analyst. Provide a concise summary (200-300 words) of this dataset analysis.

Dataset Overview:
- Rows: ${analysisData.numRows}
- Columns: ${analysisData.numCols}
- Overall missing data: ${analysisData.totalNullPercent}%

Column Details:
${analysisData.columnAnalysis.map(col => `
- ${col.name} (${col.type}): ${col.nullPercent}% missing, ${col.uniqueCount} unique values
  ${col.type === 'numeric' ? `Stats: Mean=${col.stats.mean}, Std=${col.stats.std}, Range=[${col.stats.min}, ${col.stats.max}]` : `Sample values: ${col.sampleValues.join(', ')}`}
`).join('')}

${analysisData.correlations.length > 0 ? `Notable Correlations:\n${analysisData.correlations.map(c => `- ${c.col1} ↔ ${c.col2}: ${c.correlation}`).join('\n')}` : ''}

Provide insights about data quality, patterns, and potential areas of interest for analysis.`;

        try {
            const response = await fetch('https://api.anthropic.com/v1/messages', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: 'claude-sonnet-4-20250514',
                    max_tokens: 1000,
                    messages: [{ role: 'user', content: prompt }]
                })
            });

            const data = await response.json();
            return data.content.find(c => c.type === 'text')?.text || 'Summary generation failed';
        } catch (err) {
            return `Analysis complete! The dataset contains ${analysisData.numRows} rows and ${analysisData.numCols} columns. ${analysisData.columnAnalysis.filter(c => c.type === 'numeric').length} columns are numeric, while ${analysisData.columnAnalysis.filter(c => c.type === 'categorical').length} are categorical. Missing data accounts for ${analysisData.totalNullPercent}% of total entries. ${analysisData.correlations.length > 0 ? `Notable correlations found between ${analysisData.correlations[0].col1} and ${analysisData.correlations[0].col2} (${analysisData.correlations[0].correlation}).` : 'No strong correlations detected between numeric columns.'}`;
        }
    };

    const handleFileUpload = async (e) => {
        const uploadedFile = e.target.files[0];
        if (!uploadedFile) return;

        if (!uploadedFile.name.endsWith('.csv')) {
            setError('Please upload a valid CSV file');
            return;
        }

        setFile(uploadedFile);
        setError('');
        setAnalyzing(true);
        setAnalysis(null);
        setSummary('');

        Papa.parse(uploadedFile, {
            header: true,
            skipEmptyLines: true,
            complete: async (results) => {
                try {
                    if (results.errors.length > 0) {
                        setError('CSV parsing errors detected');
                        setAnalyzing(false);
                        return;
                    }

                    if (results.data.length === 0) {
                        setError('CSV file is empty');
                        setAnalyzing(false);
                        return;
                    }

                    const analysisResult = analyzeCSV(results.data, results.meta.fields);
                    setAnalysis(analysisResult);

                    const summaryText = await generateLLMSummary(analysisResult);
                    setSummary(summaryText);
                    setAnalyzing(false);
                } catch (err) {
                    setError('Error analyzing CSV: ' + err.message);
                    setAnalyzing(false);
                }
            },
            error: (err) => {
                setError('Failed to parse CSV: ' + err.message);
                setAnalyzing(false);
            }
        });
    };

    const downloadReport = () => {
        const report = `CSV ANALYSIS REPORT
${'═'.repeat(80)}

FILE INFORMATION
${'─'.repeat(80)}
Filename: ${file.name}
Generated: ${new Date().toLocaleString()}
Analyst: CSV Analyzer v1.0

${'═'.repeat(80)}
EXECUTIVE SUMMARY
${'═'.repeat(80)}

${summary}

${'═'.repeat(80)}
DATASET METRICS
${'═'.repeat(80)}

Dimensions: ${analysis.numRows.toLocaleString()} rows × ${analysis.numCols} columns
Data Completeness: ${(100 - parseFloat(analysis.totalNullPercent)).toFixed(2)}%
Missing Values: ${analysis.totalNullPercent}%

${'═'.repeat(80)}
COLUMN-LEVEL ANALYSIS
${'═'.repeat(80)}

${analysis.columnAnalysis.map((col, idx) => `
[${idx + 1}] ${col.name}
${'─'.repeat(40)}
Type: ${col.type.toUpperCase()}
Missing: ${col.nullPercent}% (${col.nullCount} values)
Unique Values: ${col.uniqueCount}
Cardinality: ${((col.uniqueCount / analysis.numRows) * 100).toFixed(1)}%
${col.type === 'numeric' ? `
Statistical Summary:
  • Mean: ${col.stats.mean}
  • Median: ${col.stats.median}
  • Std Dev: ${col.stats.std}
  • Range: [${col.stats.min}, ${col.stats.max}]
  • Coefficient of Variation: ${(parseFloat(col.stats.std) / parseFloat(col.stats.mean) * 100).toFixed(2)}%` : `
Sample Values:
  ${col.sampleValues.slice(0, 5).map(v => `• ${v}`).join('\n  ')}`}
`).join('\n')}

${analysis.correlations.length > 0 ? `${'═'.repeat(80)}
CORRELATION ANALYSIS
${'═'.repeat(80)}

Detected ${analysis.correlations.length} significant correlation(s) (|r| > 0.3):

${analysis.correlations.map((c, idx) => `[${idx + 1}] ${c.col1} ↔ ${c.col2}
    Pearson's r: ${c.correlation}
    Strength: ${Math.abs(parseFloat(c.correlation)) > 0.7 ? 'Strong' : Math.abs(parseFloat(c.correlation)) > 0.5 ? 'Moderate' : 'Weak'}
    Direction: ${parseFloat(c.correlation) > 0 ? 'Positive' : 'Negative'}`).join('\n\n')}` : ''}

${'═'.repeat(80)}
DATA QUALITY ASSESSMENT
${'═'.repeat(80)}

Completeness Score: ${(100 - parseFloat(analysis.totalNullPercent)).toFixed(1)}%
Columns with >10% Missing: ${analysis.columnAnalysis.filter(c => parseFloat(c.nullPercent) > 10).length}
High Cardinality Columns: ${analysis.columnAnalysis.filter(c => c.uniqueCount > analysis.numRows * 0.9).length}

${'═'.repeat(80)}
End of Report
`;

        const blob = new Blob([report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${file.name.replace('.csv', '')}_analysis_report.txt`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const getDataQualityScore = () => {
        if (!analysis) return 0;
        return Math.round(100 - parseFloat(analysis.totalNullPercent));
    };

    const typeDistribution = analysis ? [
        { name: 'Numeric', value: analysis.columnAnalysis.filter(c => c.type === 'numeric').length },
        { name: 'Categorical', value: analysis.columnAnalysis.filter(c => c.type === 'categorical').length }
    ] : [];

    const COLORS = ['#06b6d4', '#3b82f6'];

    return (
        <div className="min-h-screen bg-slate-950 text-slate-100">
            {/* Top Navigation Bar */}
            <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 sm:py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 sm:gap-3">
                            <div className="w-7 h-7 sm:w-8 sm:h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center flex-shrink-0">
                                <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
                            </div>
                            <div>
                                <h1 className="text-base sm:text-lg font-semibold text-slate-100">CSV Analyzer</h1>
                                <p className="text-xs text-slate-400 hidden sm:block">Professional Data Profiling Suite</p>
                            </div>
                        </div>
                        {analysis && (
                            <button
                                onClick={downloadReport}
                                className="flex items-center gap-1 sm:gap-2 px-3 sm:px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-100 rounded-lg transition-colors border border-slate-700 text-xs sm:text-sm font-medium"
                            >
                                <Download className="w-3 h-3 sm:w-4 sm:h-4" />
                                <span className="hidden sm:inline">Export Report</span>
                                <span className="sm:hidden">Export</span>
                            </button>
                        )}
                    </div>
                </div>
            </nav>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-8">
                {/* Upload Section */}
                {!analysis && !analyzing && (
                    <div className="max-w-3xl mx-auto">
                        <div className="text-center mb-6 sm:mb-8">
                            <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 mb-2">Upload Dataset</h2>
                            <p className="text-sm sm:text-base text-slate-400">Upload a CSV file to begin automated profiling and analysis</p>
                        </div>

                        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 sm:p-8">
                            <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 sm:p-12 text-center hover:border-cyan-500/50 hover:bg-slate-800/50 transition-all">
                                <Upload className="w-10 h-10 sm:w-12 sm:h-12 mx-auto mb-4 text-slate-400" />
                                <label className="cursor-pointer">
                                    <span className="inline-flex items-center gap-2 bg-cyan-600 hover:bg-cyan-700 text-white px-5 py-2.5 sm:px-6 sm:py-3 rounded-lg font-medium transition-colors text-sm sm:text-base">
                                        <FileText className="w-4 h-4" />
                                        Select CSV File
                                    </span>
                                    <input
                                        type="file"
                                        accept=".csv"
                                        onChange={handleFileUpload}
                                        className="hidden"
                                    />
                                </label>
                                <p className="text-slate-500 text-xs sm:text-sm mt-4">Supports CSV files up to 50MB</p>
                                {file && (
                                    <div className="mt-6 inline-flex items-center gap-2 bg-slate-800 px-4 py-2 rounded-lg border border-slate-700">
                                        <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                                        <span className="text-slate-300 text-xs sm:text-sm font-medium truncate max-w-xs">{file.name}</span>
                                    </div>
                                )}
                            </div>

                            {error && (
                                <div className="mt-6 bg-red-950/50 border border-red-900/50 rounded-lg p-4 flex items-start gap-3">
                                    <XCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                                    <div>
                                        <p className="text-red-300 font-medium text-sm">Error</p>
                                        <p className="text-red-400 text-xs sm:text-sm mt-1">{error}</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Loading State */}
                {analyzing && (
                    <div className="max-w-2xl mx-auto">
                        <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 sm:p-12 text-center">
                            <Loader2 className="w-10 h-10 sm:w-12 sm:h-12 mx-auto mb-4 text-cyan-400 animate-spin" />
                            <h3 className="text-lg sm:text-xl font-semibold text-slate-200 mb-2">Processing Dataset</h3>
                            <p className="text-sm sm:text-base text-slate-400">Performing statistical analysis and generating insights...</p>
                            <div className="mt-6 flex flex-wrap justify-center gap-2 sm:gap-4 text-xs sm:text-sm text-slate-500">
                                <span>• Profiling columns</span>
                                <span>• Calculating statistics</span>
                                <span>• Detecting correlations</span>
                            </div>
                        </div>
                    </div>
                )}

                {/* Analysis Results */}
                {analysis && !analyzing && (
                    <div className="space-y-6">
                        {/* Summary Cards Row */}
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
                            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 sm:p-5">
                                <div className="flex items-center justify-between mb-2 sm:mb-3">
                                    <span className="text-slate-400 text-xs sm:text-sm font-medium">Dataset Size</span>
                                    <Database className="w-3 h-3 sm:w-4 sm:h-4 text-cyan-400" />
                                </div>
                                <div className="text-xl sm:text-2xl font-bold text-slate-100">{analysis.numRows.toLocaleString()}</div>
                                <div className="text-xs text-slate-500 mt-1">rows</div>
                            </div>

                            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 sm:p-5">
                                <div className="flex items-center justify-between mb-2 sm:mb-3">
                                    <span className="text-slate-400 text-xs sm:text-sm font-medium">Features</span>
                                    <BarChart3 className="w-3 h-3 sm:w-4 sm:h-4 text-blue-400" />
                                </div>
                                <div className="text-xl sm:text-2xl font-bold text-slate-100">{analysis.numCols}</div>
                                <div className="text-xs text-slate-500 mt-1">columns</div>
                            </div>

                            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 sm:p-5">
                                <div className="flex items-center justify-between mb-2 sm:mb-3">
                                    <span className="text-slate-400 text-xs sm:text-sm font-medium">Data Quality</span>
                                    <Activity className="w-3 h-3 sm:w-4 sm:h-4 text-emerald-400" />
                                </div>
                                <div className="text-xl sm:text-2xl font-bold text-slate-100">{getDataQualityScore()}%</div>
                                <div className="text-xs text-slate-500 mt-1">completeness</div>
                            </div>

                            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 sm:p-5">
                                <div className="flex items-center justify-between mb-2 sm:mb-3">
                                    <span className="text-slate-400 text-xs sm:text-sm font-medium">Correlations</span>
                                    <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4 text-amber-400" />
                                </div>
                                <div className="text-xl sm:text-2xl font-bold text-slate-100">{analysis.correlations.length}</div>
                                <div className="text-xs text-slate-500 mt-1">detected</div>
                            </div>
                        </div>

                        {/* Main Content Grid */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
                            {/* Left Column - Summary */}
                            <div className="lg:col-span-2 space-y-4 sm:space-y-6">
                                {/* AI Summary */}
                                <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                                    <div className="border-b border-slate-800 p-4 sm:p-5">
                                        <div className="flex items-center gap-2">
                                            <Activity className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400" />
                                            <h2 className="text-base sm:text-lg font-semibold text-slate-100">Executive Summary</h2>
                                        </div>
                                    </div>
                                    <div className="p-4 sm:p-6">
                                        <div className="prose prose-invert prose-sm max-w-none">
                                            <p className="text-sm sm:text-base text-slate-300 leading-relaxed whitespace-pre-wrap">{summary}</p>
                                        </div>
                                    </div>
                                </div>

                                {/* Column Analysis Table */}
                                <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                                    <div className="border-b border-slate-800 p-4 sm:p-5">
                                        <div className="flex items-center gap-2">
                                            <FileText className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400" />
                                            <h2 className="text-base sm:text-lg font-semibold text-slate-100">Column Profiles</h2>
                                        </div>
                                    </div>
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-xs sm:text-sm">
                                            <thead className="bg-slate-800/50">
                                                <tr>
                                                    <th className="text-left p-3 sm:p-4 font-semibold text-slate-300">Column</th>
                                                    <th className="text-left p-3 sm:p-4 font-semibold text-slate-300">Type</th>
                                                    <th className="text-right p-3 sm:p-4 font-semibold text-slate-300">Missing</th>
                                                    <th className="text-right p-3 sm:p-4 font-semibold text-slate-300 hidden sm:table-cell">Unique</th>
                                                    <th className="text-right p-3 sm:p-4 font-semibold text-slate-300 hidden md:table-cell">Cardinality</th>
                                                    <th className="text-left p-3 sm:p-4 font-semibold text-slate-300 hidden lg:table-cell">Statistics</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {analysis.columnAnalysis.map((col, idx) => (
                                                    <tr key={idx} className="border-t border-slate-800 hover:bg-slate-800/30 transition-colors">
                                                        <td className="p-3 sm:p-4 font-medium text-slate-200">{col.name}</td>
                                                        <td className="p-3 sm:p-4">
                                                            <span className={`px-2 py-1 rounded text-xs font-medium ${col.type === 'numeric'
                                                                    ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                                                                    : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                                                                }`}>
                                                                {col.type}
                                                            </span>
                                                        </td>
                                                        <td className="p-3 sm:p-4 text-right">
                                                            <span className={`font-medium ${parseFloat(col.nullPercent) > 10 ? 'text-red-400' : 'text-slate-400'}`}>
                                                                {col.nullPercent}%
                                                            </span>
                                                        </td>
                                                        <td className="p-3 sm:p-4 text-right text-slate-400 hidden sm:table-cell">{col.uniqueCount.toLocaleString()}</td>
                                                        <td className="p-3 sm:p-4 text-right text-slate-400 hidden md:table-cell">
                                                            {((col.uniqueCount / analysis.numRows) * 100).toFixed(1)}%
                                                        </td>
                                                        <td className="p-3 sm:p-4 text-slate-400 font-mono text-xs hidden lg:table-cell">
                                                            {col.type === 'numeric'
                                                                ? `μ=${col.stats.mean} σ=${col.stats.std}`
                                                                : col.sampleValues.slice(0, 2).join(', ').substring(0, 30) + '...'}
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>

                                {/* Missing Data Visualization */}
                                <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                                    <div className="border-b border-slate-800 p-4 sm:p-5">
                                        <div className="flex items-center gap-2">
                                            <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400" />
                                            <h2 className="text-base sm:text-lg font-semibold text-slate-100">Missing Value Analysis</h2>
                                        </div>
                                    </div>
                                    <div className="p-4 sm:p-6">
                                        <ResponsiveContainer width="100%" height={250}>
                                            <BarChart data={analysis.columnAnalysis}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                                <XAxis
                                                    dataKey="name"
                                                    angle={-45}
                                                    textAnchor="end"
                                                    height={80}
                                                    stroke="#64748b"
                                                    tick={{ fill: '#94a3b8', fontSize: 10 }}
                                                />
                                                <YAxis
                                                    stroke="#64748b"
                                                    tick={{ fill: '#94a3b8', fontSize: 10 }}
                                                    label={{ value: 'Missing (%)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }}
                                                />
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: '#1e293b',
                                                        border: '1px solid #334155',
                                                        borderRadius: '0.5rem',
                                                        fontSize: '11px'
                                                    }}
                                                    labelStyle={{ color: '#e2e8f0' }}
                                                />
                                                <Bar dataKey="nullPercent" radius={[4, 4, 0, 0]}>
                                                    {analysis.columnAnalysis.map((entry, index) => (
                                                        <Cell
                                                            key={`cell-${index}`}
                                                            fill={parseFloat(entry.nullPercent) > 10 ? '#ef4444' : '#06b6d4'}
                                                        />
                                                    ))}
                                                </Bar>
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>

                            {/* Right Column - Stats & Correlations */}
                            <div className="space-y-4 sm:space-y-6">
                                {/* Type Distribution */}
                                <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                                    <div className="border-b border-slate-800 p-4 sm:p-5">
                                        <h3 className="text-sm font-semibold text-slate-100">Type Distribution</h3>
                                    </div>
                                    <div className="p-4 sm:p-6">
                                        <ResponsiveContainer width="100%" height={180}>
                                            <PieChart>
                                                <Pie
                                                    data={typeDistribution}
                                                    cx="50%"
                                                    cy="50%"
                                                    innerRadius={40}
                                                    outerRadius={70}
                                                    paddingAngle={5}
                                                    dataKey="value"
                                                >
                                                    {typeDistribution.map((entry, index) => (
                                                        <Cell key={`cell-${index}`} fill={COLORS[index]} />
                                                    ))}
                                                </Pie>
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: '#1e293b',
                                                        border: '1px solid #334155',
                                                        borderRadius: '0.5rem',
                                                        fontSize: '11px'
                                                    }}
                                                />
                                            </PieChart>
                                        </ResponsiveContainer>
                                        <div className="mt-4 space-y-2">
                                            {typeDistribution.map((item, idx) => (
                                                <div key={idx} className="flex items-center justify-between text-xs sm:text-sm">
                                                    <div className="flex items-center gap-2">
                                                        <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: COLORS[idx] }}></div>
                                                        <span className="text-slate-400">{item.name}</span>
                                                    </div>
                                                    <span className="text-slate-300 font-semibold">{item.value}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>

                                {/* Correlations */}
                                {analysis.correlations.length > 0 && (
                                    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                                        <div className="border-b border-slate-800 p-4 sm:p-5">
                                            <div className="flex items-center gap-2">
                                                <TrendingUp className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400" />
                                                <h3 className="text-sm font-semibold text-slate-100">Correlations</h3>
                                            </div>
                                        </div>
                                        <div className="p-3 sm:p-4 space-y-2">
                                            {analysis.correlations.map((corr, idx) => (
                                                <div key={idx} className="bg-slate-800/50 border border-slate-700 rounded-lg p-3 sm:p-4">
                                                    <div className="flex items-center justify-between mb-2">
                                                        <span className="text-xs font-medium text-slate-400">Pearson's r</span>
                                                        <span className={`text-sm font-bold ${parseFloat(corr.correlation) > 0
                                                                ? 'text-emerald-400'
                                                                : 'text-red-400'
                                                            }`}>
                                                            {corr.correlation}
                                                        </span>
                                                    </div>
                                                    <div className="text-xs sm:text-sm text-slate-300">
                                                        <span className="text-cyan-400">{corr.col1}</span>
                                                        <span className="text-slate-600 mx-1">↔</span>
                                                        <span className="text-blue-400">{corr.col2}</span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Data Quality Metrics */}
                                <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                                    <div className="border-b border-slate-800 p-4 sm:p-5">
                                        <h3 className="text-sm font-semibold text-slate-100">Data Quality</h3>
                                    </div>
                                    <div className="p-3 sm:p-4 space-y-4">
                                        <div>
                                            <div className="flex items-center justify-between mb-2">
                                                <span className="text-xs sm:text-sm text-slate-400">Completeness</span>
                                                <span className="text-xs sm:text-sm font-semibold text-slate-200">{getDataQualityScore()}%</span>
                                            </div>
                                            <div className="w-full bg-slate-800 rounded-full h-2">
                                                <div
                                                    className="bg-gradient-to-r from-emerald-500 to-green-500 h-2 rounded-full transition-all"
                                                    style={{ width: `${getDataQualityScore()}%` }}
                                                />
                                            </div>
                                        </div>

                                        <div className="pt-4 border-t border-slate-800 space-y-3 text-xs sm:text-sm">
                                            <div className="flex items-center justify-between">
                                                <span className="text-slate-400">Complete Columns</span>
                                                <span className="text-slate-200 font-medium">
                                                    {analysis.columnAnalysis.filter(c => parseFloat(c.nullPercent) === 0).length}/{analysis.numCols}
                                                </span>
                                            </div>
                                            <div className="flex items-center justify-between">
                                                <span className="text-slate-400">High Cardinality</span>
                                                <span className="text-slate-200 font-medium">
                                                    {analysis.columnAnalysis.filter(c => c.uniqueCount > analysis.numRows * 0.9).length}
                                                </span>
                                            </div>
                                            <div className="flex items-center justify-between">
                                                <span className="text-slate-400">Numeric Features</span>
                                                <span className="text-slate-200 font-medium">
                                                    {analysis.columnAnalysis.filter(c => c.type === 'numeric').length}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default CSVSummarizer;
