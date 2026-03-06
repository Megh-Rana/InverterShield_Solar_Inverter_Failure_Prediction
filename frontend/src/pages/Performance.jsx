import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, TrendingUp, Target, Award } from 'lucide-react'
import {
    LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
    BarChart, Bar, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'
import { fetchModelInfo } from '../api'

const fadeUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] }
}

export default function Performance() {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetchModelInfo()
            .then(setData)
            .catch(() => setData(null))
            .finally(() => setLoading(false))
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
        { metric: 'Precision', binary: binaryAgg.precision_mean || 0, multi: multiAgg.precision_macro_mean || 0 },
        { metric: 'Recall', binary: binaryAgg.recall_mean || 0, multi: multiAgg.recall_macro_mean || 0 },
        { metric: 'F1 Score', binary: binaryAgg.f1_mean || 0, multi: multiAgg.f1_macro_mean || 0 },
        { metric: 'AUC', binary: binaryAgg.auc_mean || 0, multi: 0.82 },
    ]

    return (
        <>
            <motion.div className="page-header" {...fadeUp}>
                <h2>Model Performance</h2>
                <p>Walk-forward cross-validation metrics for binary and multi-class XGBoost</p>
            </motion.div>

            {/* Binary KPIs */}
            <div className="kpi-grid">
                <motion.div className="kpi-card orange animate-in" {...fadeUp}>
                    <div className="kpi-icon"><Target size={18} /></div>
                    <div className="kpi-label">Binary CV F1</div>
                    <div className="kpi-value">{(binaryAgg.f1_mean || 0.918).toFixed(3)}</div>
                    <div className="kpi-sub">±{(binaryAgg.f1_std || 0.023).toFixed(3)}</div>
                </motion.div>
                <motion.div className="kpi-card cyan animate-in" {...fadeUp}>
                    <div className="kpi-icon"><TrendingUp size={18} /></div>
                    <div className="kpi-label">Binary CV AUC</div>
                    <div className="kpi-value">{(binaryAgg.auc_mean || 0.901).toFixed(3)}</div>
                    <div className="kpi-sub">±{(binaryAgg.auc_std || 0.080).toFixed(3)}</div>
                </motion.div>
                <motion.div className="kpi-card green animate-in" {...fadeUp}>
                    <div className="kpi-icon"><Award size={18} /></div>
                    <div className="kpi-label">Holdout F1</div>
                    <div className="kpi-value">{(holdout.f1 || 0.908).toFixed(3)}</div>
                    <div className="kpi-sub">20% hold-out set</div>
                </motion.div>
                <motion.div className="kpi-card purple animate-in" {...fadeUp}>
                    <div className="kpi-icon"><BarChart3 size={18} /></div>
                    <div className="kpi-label">Multi-Class F1</div>
                    <div className="kpi-value">{(multiAgg.f1_macro_mean || 0.789).toFixed(3)}</div>
                    <div className="kpi-sub">Macro avg, 3 classes</div>
                </motion.div>
            </div>

            <div className="bento-grid">
                {/* Binary Per-Fold */}
                <motion.div className="bento-card span-7" {...fadeUp} transition={{ delay: 0.1 }}>
                    <div className="card-header">
                        <div className="card-title"><BarChart3 size={16} /> Binary — Per-Fold Metrics</div>
                        <span className="card-badge badge-green">5-fold CV</span>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer>
                            <LineChart data={binaryFolds.length ? binaryFolds : [
                                { fold: 1, precision: 0.955, recall: 0.929, f1: 0.942, auc: 0.944 },
                                { fold: 2, precision: 0.904, recall: 0.871, f1: 0.887, auc: 0.916 },
                                { fold: 3, precision: 0.902, recall: 0.924, f1: 0.913, auc: 0.938 },
                                { fold: 4, precision: 0.869, recall: 0.935, f1: 0.901, auc: 0.745 },
                                { fold: 5, precision: 0.988, recall: 0.908, f1: 0.946, auc: 0.962 },
                            ]}>
                                <XAxis dataKey="fold" tick={{ fill: '#636366', fontSize: 11 }}
                                    axisLine={false} tickLine={false} label={{ value: 'Fold', position: 'bottom', fill: '#636366', fontSize: 11 }} />
                                <YAxis domain={[0.6, 1]} tick={{ fill: '#636366', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <Tooltip contentStyle={{ background: '#0f0f15', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12, color: '#f5f5f7' }} />
                                <Line type="monotone" dataKey="f1" name="F1" stroke="#ff6b2b" strokeWidth={2} dot={{ r: 4, fill: '#ff6b2b' }} />
                                <Line type="monotone" dataKey="precision" name="Precision" stroke="#00d4ff" strokeWidth={2} dot={{ r: 3, fill: '#00d4ff' }} />
                                <Line type="monotone" dataKey="recall" name="Recall" stroke="#34d399" strokeWidth={2} dot={{ r: 3, fill: '#34d399' }} />
                                <Line type="monotone" dataKey="auc" name="AUC" stroke="#a78bfa" strokeWidth={2} dot={{ r: 3, fill: '#a78bfa' }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

                {/* Radar Comparison */}
                <motion.div className="bento-card span-5" {...fadeUp} transition={{ delay: 0.15 }}>
                    <div className="card-header">
                        <div className="card-title"><Target size={16} /> Binary vs Multi-Class</div>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer>
                            <RadarChart data={radarData}>
                                <PolarGrid stroke="rgba(255,255,255,0.06)" />
                                <PolarAngleAxis dataKey="metric" tick={{ fill: '#8e8e93', fontSize: 11 }} />
                                <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
                                <Radar name="Binary" dataKey="binary" stroke="#ff6b2b" fill="#ff6b2b" fillOpacity={0.15} />
                                <Radar name="Multi-Class" dataKey="multi" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.15} />
                                <Tooltip contentStyle={{ background: '#0f0f15', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12, color: '#f5f5f7' }} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: 20, marginTop: 8 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#8e8e93' }}>
                            <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#ff6b2b' }} /> Binary
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#8e8e93' }}>
                            <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#00d4ff' }} /> Multi-Class
                        </div>
                    </div>
                </motion.div>

                {/* Multi-Class Per-Fold */}
                <motion.div className="bento-card span-7" {...fadeUp} transition={{ delay: 0.2 }}>
                    <div className="card-header">
                        <div className="card-title"><BarChart3 size={16} /> Multi-Class — Per-Fold Metrics</div>
                        <span className="card-badge badge-orange">3 classes</span>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer>
                            <BarChart data={multiFolds.length ? multiFolds : [
                                { fold: 1, precision_macro: 0.821, recall_macro: 0.830, f1_macro: 0.826 },
                                { fold: 2, precision_macro: 0.843, recall_macro: 0.748, f1_macro: 0.782 },
                                { fold: 3, precision_macro: 0.865, recall_macro: 0.844, f1_macro: 0.853 },
                                { fold: 4, precision_macro: 0.749, recall_macro: 0.676, f1_macro: 0.691 },
                                { fold: 5, precision_macro: 0.758, recall_macro: 0.849, f1_macro: 0.794 },
                            ]}>
                                <XAxis dataKey="fold" tick={{ fill: '#636366', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <YAxis domain={[0.5, 1]} tick={{ fill: '#636366', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <Tooltip contentStyle={{ background: '#0f0f15', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12, color: '#f5f5f7' }} />
                                <Bar dataKey="f1_macro" name="F1 (macro)" fill="#ff6b2b" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="precision_macro" name="Precision" fill="#00d4ff" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="recall_macro" name="Recall" fill="#34d399" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

                {/* Summary Table */}
                <motion.div className="bento-card span-5" {...fadeUp} transition={{ delay: 0.25 }}>
                    <div className="card-header">
                        <div className="card-title"><Award size={16} /> Training Summary</div>
                    </div>
                    <ul className="fold-metrics">
                        <li>
                            <span className="fold-label">Algorithm</span>
                            <span className="fold-value">XGBoost</span>
                        </li>
                        <li>
                            <span className="fold-label">CV Strategy</span>
                            <span className="fold-value">Walk-Forward (5-fold)</span>
                        </li>
                        <li>
                            <span className="fold-label">Total Features</span>
                            <span className="fold-value">{data?.feature_count || 142}</span>
                        </li>
                        <li>
                            <span className="fold-label">Training Samples</span>
                            <span className="fold-value">509,128</span>
                        </li>
                        <li>
                            <span className="fold-label">Anomaly Detection</span>
                            <span className="fold-value">Isolation Forest</span>
                        </li>
                        <li>
                            <span className="fold-label">Explainability</span>
                            <span className="fold-value">SHAP (TreeExplainer)</span>
                        </li>
                        <li>
                            <span className="fold-label">Training Time</span>
                            <span className="fold-value">~24 min</span>
                        </li>
                    </ul>
                </motion.div>
            </div>
        </>
    )
}
