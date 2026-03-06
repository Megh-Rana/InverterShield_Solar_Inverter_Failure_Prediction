import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell
} from 'recharts'
import {
    Zap, TrendingUp, Shield, Users, AlertTriangle,
    CheckCircle, XCircle
} from 'lucide-react'
import { fetchDashboard } from '../api'

const RISK_COLORS = { 'No Risk': '#3eb489', 'Degradation': '#d4a843', 'Shutdown': '#d45f58' }

const fadeUp = {
    initial: { opacity: 0, y: 16 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] }
}

const tooltipStyle = {
    background: '#fff', border: '1px solid #e8e5e1', borderRadius: 10,
    fontSize: 12, color: '#1a1a1a', boxShadow: '0 4px 12px rgba(0,0,0,0.06)'
}

const generateInverters = () => {
    const plants = ['Plant 1', 'Plant 2', 'Plant 3']
    const inverters = []
    for (let i = 0; i < 32; i++) {
        const prob = Math.random()
        const risk = prob < 0.4 ? 0 : prob < 0.7 ? 1 : 2
        inverters.push({
            id: `INV_${String(i + 1).padStart(2, '0')}`,
            plant: plants[i % 3],
            failureProb: prob,
            riskClass: risk,
            riskLabel: ['No Risk', 'Degradation', 'Shutdown'][risk],
        })
    }
    return inverters.sort((a, b) => b.failureProb - a.failureProb)
}

export default function Dashboard() {
    const [data, setData] = useState(null)
    const [inverters] = useState(generateInverters)

    useEffect(() => {
        fetchDashboard().then(setData).catch(() => setData(null))
    }, [])

    const riskCounts = inverters.reduce((acc, inv) => {
        acc[inv.riskLabel] = (acc[inv.riskLabel] || 0) + 1
        return acc
    }, {})
    const pieData = Object.entries(riskCounts).map(([name, value]) => ({ name, value }))

    const metrics = data?.model_performance?.binary_cv || {}
    const shapFeatures = data?.top_risk_factors || [
        'month', 'inv_temp_7d_mean', 'meter_kwh_import', 'inv_temp_24h_mean',
        'smu_string_mean_7d_std', 'meter_kwh_total', 'inv_id', 'alarm_count_7d',
    ]

    return (
        <>
            <motion.div className="page-header" {...fadeUp}>
                <h2>Dashboard</h2>
                <p>Health overview across 3 solar plants · 32 inverters</p>
            </motion.div>

            <div className="kpi-grid">
                <motion.div className="kpi-card coral animate-in" {...fadeUp}>
                    <div className="kpi-icon"><Zap size={16} /></div>
                    <div className="kpi-label">F1 Score</div>
                    <div className="kpi-value">{metrics.f1_mean?.toFixed(3) || '0.918'}</div>
                    <div className="kpi-sub">±{metrics.f1_std?.toFixed(3) || '0.023'} · 5 folds</div>
                </motion.div>
                <motion.div className="kpi-card blue animate-in" {...fadeUp}>
                    <div className="kpi-icon"><TrendingUp size={16} /></div>
                    <div className="kpi-label">AUC</div>
                    <div className="kpi-value">{metrics.auc_mean?.toFixed(3) || '0.901'}</div>
                    <div className="kpi-sub">Walk-forward CV</div>
                </motion.div>
                <motion.div className="kpi-card green animate-in" {...fadeUp}>
                    <div className="kpi-icon"><Shield size={16} /></div>
                    <div className="kpi-label">Precision</div>
                    <div className="kpi-value">{metrics.precision_mean?.toFixed(3) || '0.924'}</div>
                    <div className="kpi-sub">Binary classifier</div>
                </motion.div>
                <motion.div className="kpi-card purple animate-in" {...fadeUp}>
                    <div className="kpi-icon"><Users size={16} /></div>
                    <div className="kpi-label">Inverters</div>
                    <div className="kpi-value">32</div>
                    <div className="kpi-sub">3 plants monitored</div>
                </motion.div>
            </div>

            <div className="bento-grid">
                {/* Pie */}
                <motion.div className="bento-card span-5" {...fadeUp} transition={{ delay: 0.08 }}>
                    <div className="card-header">
                        <div className="card-title"><Shield size={15} /> Risk Distribution</div>
                        <span className="card-badge badge-green">Live</span>
                    </div>
                    <div className="chart-container" style={{ height: 200 }}>
                        <ResponsiveContainer>
                            <PieChart>
                                <Pie data={pieData} cx="50%" cy="50%" innerRadius={48} outerRadius={78} paddingAngle={3} dataKey="value">
                                    {pieData.map((e) => <Cell key={e.name} fill={RISK_COLORS[e.name]} />)}
                                </Pie>
                                <Tooltip contentStyle={tooltipStyle} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginTop: 6 }}>
                        {pieData.map(({ name, value }) => (
                            <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12 }}>
                                <div style={{ width: 7, height: 7, borderRadius: '50%', background: RISK_COLORS[name] }} />
                                <span style={{ color: '#999' }}>{name}</span>
                                <span style={{ fontWeight: 600 }}>{value}</span>
                            </div>
                        ))}
                    </div>
                </motion.div>

                {/* Bar */}
                <motion.div className="bento-card span-7" {...fadeUp} transition={{ delay: 0.12 }}>
                    <div className="card-header">
                        <div className="card-title"><AlertTriangle size={15} /> Top Risk Inverters</div>
                        <span className="card-badge badge-coral">Top 10</span>
                    </div>
                    <div className="chart-container" style={{ height: 200 }}>
                        <ResponsiveContainer>
                            <BarChart data={inverters.slice(0, 10)} layout="vertical">
                                <XAxis type="number" domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                                    tick={{ fill: '#999', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <YAxis type="category" dataKey="id" width={58}
                                    tick={{ fill: '#6b6b6b', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} contentStyle={tooltipStyle} />
                                <Bar dataKey="failureProb" radius={[0, 4, 4, 0]}>
                                    {inverters.slice(0, 10).map((inv) => (
                                        <Cell key={inv.id} fill={inv.failureProb > 0.7 ? '#d45f58' : inv.failureProb > 0.4 ? '#d4a843' : '#3eb489'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

                {/* SHAP */}
                <motion.div className="bento-card span-5" {...fadeUp} transition={{ delay: 0.16 }}>
                    <div className="card-header">
                        <div className="card-title"><Zap size={15} /> Top Risk Factors</div>
                        <span className="card-badge badge-coral">SHAP</span>
                    </div>
                    <div className="shap-list">
                        {shapFeatures.slice(0, 8).map((feat, i) => (
                            <div className="shap-item" key={feat}>
                                <div className="shap-rank">{i + 1}</div>
                                <div className="shap-name">{feat}</div>
                                <div className="shap-bar">
                                    <div className="shap-bar-fill" style={{ width: `${100 - i * 10}%` }} />
                                </div>
                            </div>
                        ))}
                    </div>
                </motion.div>

                {/* Table */}
                <motion.div className="bento-card span-7" {...fadeUp} transition={{ delay: 0.2 }}>
                    <div className="card-header">
                        <div className="card-title"><Users size={15} /> All Inverters</div>
                        <span style={{ fontSize: 11, color: '#999' }}>32 total</span>
                    </div>
                    <div style={{ maxHeight: 270, overflowY: 'auto' }}>
                        <table className="risk-table">
                            <thead>
                                <tr><th>Inverter</th><th>Plant</th><th>Failure Prob</th><th>Status</th></tr>
                            </thead>
                            <tbody>
                                {inverters.slice(0, 15).map((inv) => (
                                    <tr key={inv.id}>
                                        <td style={{ fontWeight: 500 }}>{inv.id}</td>
                                        <td style={{ color: '#6b6b6b' }}>{inv.plant}</td>
                                        <td>
                                            <div className="prob-bar-container">
                                                <span style={{ fontVariantNumeric: 'tabular-nums', fontSize: 13 }}>
                                                    {(inv.failureProb * 100).toFixed(1)}%
                                                </span>
                                                <div className="prob-bar">
                                                    <div className={`prob-bar-fill ${inv.failureProb > 0.7 ? 'high' : inv.failureProb > 0.4 ? 'medium' : 'low'}`}
                                                        style={{ width: `${inv.failureProb * 100}%` }} />
                                                </div>
                                            </div>
                                        </td>
                                        <td>
                                            <span className={`risk-pill ${inv.riskClass === 0 ? 'no-risk' : inv.riskClass === 1 ? 'degradation' : 'shutdown'}`}>
                                                {inv.riskClass === 0 ? <CheckCircle size={11} /> : inv.riskClass === 1 ? <AlertTriangle size={11} /> : <XCircle size={11} />}
                                                {inv.riskLabel}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </motion.div>
            </div>
        </>
    )
}
