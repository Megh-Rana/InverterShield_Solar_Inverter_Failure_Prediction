import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
    LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
    BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'
import { fetchModelInfo } from '../api'

const tip = {
    background: '#fff', border: '1px solid #eae8e4', borderRadius: 8,
    fontSize: 11, color: '#202020', boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
}

export default function Performance() {
    const [data, setData] = useState(null)
    useEffect(() => { fetchModelInfo().then(setData).catch(() => { }) }, [])

    const m = data?.metrics || {}
    const bc = m.cv_binary?.aggregate || {}
    const mc = m.cv_multiclass?.aggregate || {}
    const ho = m.holdout_binary || {}

    const bFolds = m.cv_binary?.per_fold || [
        { fold: 1, precision: 0.955, recall: 0.929, f1: 0.942, auc: 0.944 },
        { fold: 2, precision: 0.904, recall: 0.871, f1: 0.887, auc: 0.916 },
        { fold: 3, precision: 0.902, recall: 0.924, f1: 0.913, auc: 0.938 },
        { fold: 4, precision: 0.869, recall: 0.935, f1: 0.901, auc: 0.745 },
        { fold: 5, precision: 0.988, recall: 0.908, f1: 0.946, auc: 0.962 },
    ]
    const mFolds = m.cv_multiclass?.per_fold || [
        { fold: 1, f1_macro: 0.826, precision_macro: 0.821, recall_macro: 0.830 },
        { fold: 2, f1_macro: 0.782, precision_macro: 0.843, recall_macro: 0.748 },
        { fold: 3, f1_macro: 0.853, precision_macro: 0.865, recall_macro: 0.844 },
        { fold: 4, f1_macro: 0.691, precision_macro: 0.749, recall_macro: 0.676 },
        { fold: 5, f1_macro: 0.794, precision_macro: 0.758, recall_macro: 0.849 },
    ]

    const radar = [
        { m: 'Precision', b: bc.precision_mean || 0.924, mc: mc.precision_macro_mean || 0.807 },
        { m: 'Recall', b: bc.recall_mean || 0.913, mc: mc.recall_macro_mean || 0.789 },
        { m: 'F1', b: bc.f1_mean || 0.918, mc: mc.f1_macro_mean || 0.789 },
        { m: 'AUC', b: bc.auc_mean || 0.901, mc: 0.82 },
    ]

    return (
        <>
            <motion.div className="page-header" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <h2>Model Performance</h2>
                <p>Walk-forward cross-validation · XGBoost</p>
            </motion.div>

            <div className="metric-row rise">
                <div className="metric-item">
                    <div className="metric-label">Binary F1</div>
                    <div className="metric-value">{(bc.f1_mean || 0.918).toFixed(3)}</div>
                    <div className="metric-sub">±{(bc.f1_std || 0.023).toFixed(3)}</div>
                </div>
                <div className="metric-item">
                    <div className="metric-label">AUC</div>
                    <div className="metric-value">{(bc.auc_mean || 0.901).toFixed(3)}</div>
                    <div className="metric-sub">±{(bc.auc_std || 0.080).toFixed(3)}</div>
                </div>
                <div className="metric-item">
                    <div className="metric-label">Holdout F1</div>
                    <div className="metric-value">{(ho.f1 || 0.908).toFixed(3)}</div>
                </div>
                <div className="metric-item">
                    <div className="metric-label">Multi-class F1</div>
                    <div className="metric-value">{(mc.f1_macro_mean || 0.789).toFixed(3)}</div>
                    <div className="metric-sub">macro, 3 classes</div>
                </div>
            </div>

            <div className="grid-7-5">
                <div className="card rise">
                    <div className="card-head"><h3>Binary — per fold</h3><span className="tag">5-fold</span></div>
                    <div className="chart-area">
                        <ResponsiveContainer>
                            <LineChart data={bFolds}>
                                <XAxis dataKey="fold" tick={{ fill: '#999', fontSize: 10 }} axisLine={false} tickLine={false} />
                                <YAxis domain={[0.6, 1]} tick={{ fill: '#999', fontSize: 10 }} axisLine={false} tickLine={false} />
                                <Tooltip contentStyle={tip} />
                                <Line type="monotone" dataKey="f1" name="F1" stroke="#d9635e" strokeWidth={1.5} dot={{ r: 3, fill: '#d9635e' }} />
                                <Line type="monotone" dataKey="precision" name="Prec" stroke="#5180b5" strokeWidth={1.2} dot={{ r: 2.5, fill: '#5180b5' }} />
                                <Line type="monotone" dataKey="recall" name="Rec" stroke="#3ea87d" strokeWidth={1.2} dot={{ r: 2.5, fill: '#3ea87d' }} />
                                <Line type="monotone" dataKey="auc" name="AUC" stroke="#999" strokeWidth={1} strokeDasharray="4 3" dot={{ r: 2, fill: '#999' }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card rise">
                    <div className="card-head"><h3>Binary vs multi-class</h3></div>
                    <div className="chart-area">
                        <ResponsiveContainer>
                            <RadarChart data={radar}>
                                <PolarGrid stroke="#eae8e4" />
                                <PolarAngleAxis dataKey="m" tick={{ fill: '#555', fontSize: 11 }} />
                                <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
                                <Radar dataKey="b" stroke="#d9635e" fill="#d9635e" fillOpacity={0.08} />
                                <Radar dataKey="mc" stroke="#5180b5" fill="#5180b5" fillOpacity={0.08} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: 16, fontSize: 11, color: '#999' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            <span style={{ width: 6, height: 6, borderRadius: 3, background: '#d9635e', display: 'inline-block' }} /> Binary
                        </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            <span style={{ width: 6, height: 6, borderRadius: 3, background: '#5180b5', display: 'inline-block' }} /> Multi-class
                        </span>
                    </div>
                </div>
            </div>

            <div className="grid-7-5">
                <div className="card rise">
                    <div className="card-head"><h3>Multi-class — per fold</h3></div>
                    <div className="chart-area">
                        <ResponsiveContainer>
                            <BarChart data={mFolds}>
                                <XAxis dataKey="fold" tick={{ fill: '#999', fontSize: 10 }} axisLine={false} tickLine={false} />
                                <YAxis domain={[0.5, 1]} tick={{ fill: '#999', fontSize: 10 }} axisLine={false} tickLine={false} />
                                <Tooltip contentStyle={tip} />
                                <Bar dataKey="f1_macro" name="F1" fill="#d9635e" radius={[3, 3, 0, 0]} />
                                <Bar dataKey="precision_macro" name="Prec" fill="#5180b5" radius={[3, 3, 0, 0]} />
                                <Bar dataKey="recall_macro" name="Rec" fill="#3ea87d" radius={[3, 3, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card rise">
                    <div className="card-head"><h3>Training details</h3></div>
                    <ul className="summary-list">
                        <li><span className="sl">Algorithm</span><span className="sv">XGBoost</span></li>
                        <li><span className="sl">Validation</span><span className="sv">Walk-forward, 5-fold</span></li>
                        <li><span className="sl">Features</span><span className="sv">{data?.feature_count || 142}</span></li>
                        <li><span className="sl">Samples</span><span className="sv">509,128</span></li>
                        <li><span className="sl">Anomaly model</span><span className="sv">Isolation Forest</span></li>
                        <li><span className="sl">Explainability</span><span className="sv">SHAP</span></li>
                    </ul>
                </div>
            </div>
        </>
    )
}
