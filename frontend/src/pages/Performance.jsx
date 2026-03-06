import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, TrendingUp, Target, Award } from 'lucide-react'
import {
    LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
    BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'
import { fetchModelInfo } from '../api'

const fadeUp = {
    initial: { opacity: 0, y: 16 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] }
}

const tooltipStyle = {
    background: '#fff', border: '1px solid #e8e5e1', borderRadius: 10,
    fontSize: 12, color: '#1a1a1a', boxShadow: '0 4px 12px rgba(0,0,0,0.06)'
}

export default function Performance() {
    const [data, setData] = useState(null)

    useEffect(() => {
        fetchModelInfo().then(setData).catch(() => setData(null))
    }, [])

    const metrics = data?.metrics || {}
    const binaryCv = metrics.cv_binary || {}
    const multiCv = metrics.cv_multiclass || {}
    const holdout = metrics.holdout_binary || {}
    const binaryFolds = binaryCv.per_fold || []
    const multiFolds = multiCv.per_fold || []
    const binaryAgg = binaryCv.aggregate || {}
    const multiAgg = multiCv.aggregate || {}

    const radarData = [
        { metric: 'Precision', binary: binaryAgg.precision_mean || 0.924, multi: multiAgg.precision_macro_mean || 0.807 },
        { metric: 'Recall', binary: binaryAgg.recall_mean || 0.913, multi: multiAgg.recall_macro_mean || 0.789 },
        { metric: 'F1 Score', binary: binaryAgg.f1_mean || 0.918, multi: multiAgg.f1_macro_mean || 0.789 },
        { metric: 'AUC', binary: binaryAgg.auc_mean || 0.901, multi: 0.82 },
    ]

    const defaultBinaryFolds = [
        { fold: 1, precision: 0.955, recall: 0.929, f1: 0.942, auc: 0.944 },
        { fold: 2, precision: 0.904, recall: 0.871, f1: 0.887, auc: 0.916 },
        { fold: 3, precision: 0.902, recall: 0.924, f1: 0.913, auc: 0.938 },
        { fold: 4, precision: 0.869, recall: 0.935, f1: 0.901, auc: 0.745 },
        { fold: 5, precision: 0.988, recall: 0.908, f1: 0.946, auc: 0.962 },
    ]

    const defaultMultiFolds = [
        { fold: 1, f1_macro: 0.826, precision_macro: 0.821, recall_macro: 0.830 },
        { fold: 2, f1_macro: 0.782, precision_macro: 0.843, recall_macro: 0.748 },
        { fold: 3, f1_macro: 0.853, precision_macro: 0.865, recall_macro: 0.844 },
        { fold: 4, f1_macro: 0.691, precision_macro: 0.749, recall_macro: 0.676 },
        { fold: 5, f1_macro: 0.794, precision_macro: 0.758, recall_macro: 0.849 },
    ]

    return (
        <>
            <motion.div className="page-header" {...fadeUp}>
                <h2>Model Performance</h2>
                <p>Walk-forward cross-validation results for XGBoost binary and multi-class</p>
            </motion.div>

            <div className="kpi-grid">
                <motion.div className="kpi-card coral animate-in" {...fadeUp}>
                    <div className="kpi-icon"><Target size={16} /></div>
                    <div className="kpi-label">Binary F1</div>
                    <div className="kpi-value">{(binaryAgg.f1_mean || 0.918).toFixed(3)}</div>
                    <div className="kpi-sub">±{(binaryAgg.f1_std || 0.023).toFixed(3)}</div>
                </motion.div>
                <motion.div className="kpi-card blue animate-in" {...fadeUp}>
                    <div className="kpi-icon"><TrendingUp size={16} /></div>
                    <div className="kpi-label">Binary AUC</div>
                    <div className="kpi-value">{(binaryAgg.auc_mean || 0.901).toFixed(3)}</div>
                    <div className="kpi-sub">±{(binaryAgg.auc_std || 0.080).toFixed(3)}</div>
                </motion.div>
                <motion.div className="kpi-card green animate-in" {...fadeUp}>
                    <div className="kpi-icon"><Award size={16} /></div>
                    <div className="kpi-label">Holdout F1</div>
                    <div className="kpi-value">{(holdout.f1 || 0.908).toFixed(3)}</div>
                    <div className="kpi-sub">20% hold-out</div>
                </motion.div>
                <motion.div className="kpi-card purple animate-in" {...fadeUp}>
                    <div className="kpi-icon"><BarChart3 size={16} /></div>
                    <div className="kpi-label">Multi-Class F1</div>
                    <div className="kpi-value">{(multiAgg.f1_macro_mean || 0.789).toFixed(3)}</div>
                    <div className="kpi-sub">Macro · 3 classes</div>
                </motion.div>
            </div>

            <div className="bento-grid">
                <motion.div className="bento-card span-7" {...fadeUp} transition={{ delay: 0.08 }}>
                    <div className="card-header">
                        <div className="card-title"><BarChart3 size={15} /> Binary — Per-Fold</div>
                        <span className="card-badge badge-green">5-fold</span>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer>
                            <LineChart data={binaryFolds.length ? binaryFolds : defaultBinaryFolds}>
                                <XAxis dataKey="fold" tick={{ fill: '#999', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <YAxis domain={[0.6, 1]} tick={{ fill: '#999', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Line type="monotone" dataKey="f1" name="F1" stroke="#e8736c" strokeWidth={2} dot={{ r: 3.5, fill: '#e8736c' }} />
                                <Line type="monotone" dataKey="precision" name="Precision" stroke="#5b8cbe" strokeWidth={1.5} dot={{ r: 3, fill: '#5b8cbe' }} />
                                <Line type="monotone" dataKey="recall" name="Recall" stroke="#3eb489" strokeWidth={1.5} dot={{ r: 3, fill: '#3eb489' }} />
                                <Line type="monotone" dataKey="auc" name="AUC" stroke="#8b7ec8" strokeWidth={1.5} dot={{ r: 3, fill: '#8b7ec8' }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

                <motion.div className="bento-card span-5" {...fadeUp} transition={{ delay: 0.12 }}>
                    <div className="card-header">
                        <div className="card-title"><Target size={15} /> Binary vs Multi-Class</div>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer>
                            <RadarChart data={radarData}>
                                <PolarGrid stroke="#e8e5e1" />
                                <PolarAngleAxis dataKey="metric" tick={{ fill: '#6b6b6b', fontSize: 11 }} />
                                <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
                                <Radar name="Binary" dataKey="binary" stroke="#e8736c" fill="#e8736c" fillOpacity={0.12} />
                                <Radar name="Multi" dataKey="multi" stroke="#5b8cbe" fill="#5b8cbe" fillOpacity={0.12} />
                                <Tooltip contentStyle={tooltipStyle} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: 20, marginTop: 4 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: '#6b6b6b' }}>
                            <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#e8736c' }} /> Binary
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: '#6b6b6b' }}>
                            <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#5b8cbe' }} /> Multi-Class
                        </div>
                    </div>
                </motion.div>

                <motion.div className="bento-card span-7" {...fadeUp} transition={{ delay: 0.16 }}>
                    <div className="card-header">
                        <div className="card-title"><BarChart3 size={15} /> Multi-Class — Per-Fold</div>
                        <span className="card-badge badge-coral">3 classes</span>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer>
                            <BarChart data={multiFolds.length ? multiFolds : defaultMultiFolds}>
                                <XAxis dataKey="fold" tick={{ fill: '#999', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <YAxis domain={[0.5, 1]} tick={{ fill: '#999', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Bar dataKey="f1_macro" name="F1 (macro)" fill="#e8736c" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="precision_macro" name="Precision" fill="#5b8cbe" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="recall_macro" name="Recall" fill="#3eb489" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

                <motion.div className="bento-card span-5" {...fadeUp} transition={{ delay: 0.2 }}>
                    <div className="card-header">
                        <div className="card-title"><Award size={15} /> Training Summary</div>
                    </div>
                    <ul className="fold-metrics">
                        <li><span className="fold-label">Algorithm</span><span className="fold-value">XGBoost</span></li>
                        <li><span className="fold-label">CV Strategy</span><span className="fold-value">Walk-Forward (5-fold)</span></li>
                        <li><span className="fold-label">Features</span><span className="fold-value">{data?.feature_count || 142}</span></li>
                        <li><span className="fold-label">Training Samples</span><span className="fold-value">509,128</span></li>
                        <li><span className="fold-label">Anomaly Detection</span><span className="fold-value">Isolation Forest</span></li>
                        <li><span className="fold-label">Explainability</span><span className="fold-value">SHAP TreeExplainer</span></li>
                    </ul>
                </motion.div>
            </div>
        </>
    )
}
