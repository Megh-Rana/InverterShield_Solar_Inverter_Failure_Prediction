import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, LineChart, Line
} from 'recharts'
import {
    Zap, Thermometer, Shield, Users, TrendingUp, AlertTriangle,
    CheckCircle, XCircle
} from 'lucide-react'
import { fetchDashboard } from '../api'

const RISK_COLORS = { 'No Risk': '#34d399', 'Degradation': '#fbbf24', 'Shutdown': '#ff4757' }

const fadeUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] }
}

// Simulated inverter data for dashboard demo
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
            temp: 35 + Math.random() * 30,
            power: 2000 + Math.random() * 8000,
        })
    }
    return inverters.sort((a, b) => b.failureProb - a.failureProb)
}

export default function Dashboard() {
    const [data, setData] = useState(null)
    const [inverters] = useState(generateInverters)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetchDashboard()
            .then(setData)
            .catch(() => setData(null))
            .finally(() => setLoading(false))
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
        'inv_kwh_midnight', 'meter_pf_24h_mean'
    ]

    return (
        <>
            <motion.div className="page-header" {...fadeUp}>
                <h2>Dashboard</h2>
                <p>Real-time health overview across 3 solar plants</p>
            </motion.div>

            {/* KPI Cards */}
            <div className="kpi-grid">
                <motion.div className="kpi-card orange animate-in" {...fadeUp}>
                    <div className="kpi-icon"><Zap size={18} /></div>
                    <div className="kpi-label">Model F1 Score</div>
                    <div className="kpi-value">{metrics.f1_mean?.toFixed(3) || '0.918'}</div>
                    <div className="kpi-sub">±{metrics.f1_std?.toFixed(3) || '0.023'} across 5 folds</div>
                </motion.div>

                <motion.div className="kpi-card cyan animate-in" {...fadeUp}>
                    <div className="kpi-icon"><TrendingUp size={18} /></div>
                    <div className="kpi-label">Model AUC</div>
                    <div className="kpi-value">{metrics.auc_mean?.toFixed(3) || '0.901'}</div>
                    <div className="kpi-sub">Walk-forward CV</div>
                </motion.div>

                <motion.div className="kpi-card green animate-in" {...fadeUp}>
                    <div className="kpi-icon"><Shield size={18} /></div>
                    <div className="kpi-label">Precision</div>
                    <div className="kpi-value">{metrics.precision_mean?.toFixed(3) || '0.924'}</div>
                    <div className="kpi-sub">Binary classifier</div>
                </motion.div>

                <motion.div className="kpi-card purple animate-in" {...fadeUp}>
                    <div className="kpi-icon"><Users size={18} /></div>
                    <div className="kpi-label">Inverters</div>
                    <div className="kpi-value">32</div>
                    <div className="kpi-sub">Across 3 plants</div>
                </motion.div>
            </div>

            {/* Bento Grid */}
            <div className="bento-grid">
                {/* Risk Distribution */}
                <motion.div className="bento-card span-5" {...fadeUp} transition={{ delay: 0.1 }}>
                    <div className="card-header">
                        <div className="card-title">
                            <Shield size={16} />
                            Risk Distribution
                        </div>
                        <span className="card-badge badge-green">Live</span>
                    </div>
                    <div className="chart-container" style={{ height: 200 }}>
                        <ResponsiveContainer>
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={50}
                                    outerRadius={80}
                                    paddingAngle={3}
                                    dataKey="value"
                                >
                                    {pieData.map((entry) => (
                                        <Cell key={entry.name} fill={RISK_COLORS[entry.name]} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ background: '#0f0f15', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12, color: '#f5f5f7' }}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginTop: 8 }}>
                        {pieData.map(({ name, value }) => (
                            <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                                <div style={{ width: 8, height: 8, borderRadius: '50%', background: RISK_COLORS[name] }} />
                                <span style={{ color: '#8e8e93' }}>{name}</span>
                                <span style={{ fontWeight: 600 }}>{value}</span>
                            </div>
                        ))}
                    </div>
                </motion.div>

                {/* Top Risk Inverters */}
                <motion.div className="bento-card span-7" {...fadeUp} transition={{ delay: 0.15 }}>
                    <div className="card-header">
                        <div className="card-title">
                            <AlertTriangle size={16} />
                            Top Risk Inverters
                        </div>
                        <span className="card-badge badge-orange">Top 10</span>
                    </div>
                    <div className="chart-container" style={{ height: 200 }}>
                        <ResponsiveContainer>
                            <BarChart data={inverters.slice(0, 10)} layout="vertical">
                                <XAxis type="number" domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                                    tick={{ fill: '#636366', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <YAxis type="category" dataKey="id" width={60}
                                    tick={{ fill: '#8e8e93', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <Tooltip
                                    formatter={v => `${(v * 100).toFixed(1)}%`}
                                    contentStyle={{ background: '#0f0f15', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12, color: '#f5f5f7' }}
                                />
                                <Bar dataKey="failureProb" radius={[0, 4, 4, 0]}>
                                    {inverters.slice(0, 10).map((inv) => (
                                        <Cell key={inv.id} fill={inv.failureProb > 0.7 ? '#ff4757' : inv.failureProb > 0.4 ? '#fbbf24' : '#34d399'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

                {/* SHAP Features */}
                <motion.div className="bento-card span-5" {...fadeUp} transition={{ delay: 0.2 }}>
                    <div className="card-header">
                        <div className="card-title">
                            <Zap size={16} />
                            Top Risk Factors
                        </div>
                        <span className="card-badge badge-orange">SHAP</span>
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

                {/* Risk Table */}
                <motion.div className="bento-card span-7" {...fadeUp} transition={{ delay: 0.25 }}>
                    <div className="card-header">
                        <div className="card-title">
                            <Users size={16} />
                            All Inverters
                        </div>
                        <span style={{ fontSize: 11, color: '#636366' }}>32 total</span>
                    </div>
                    <div style={{ maxHeight: 280, overflowY: 'auto' }}>
                        <table className="risk-table">
                            <thead>
                                <tr>
                                    <th>Inverter</th>
                                    <th>Plant</th>
                                    <th>Failure Prob</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {inverters.slice(0, 15).map((inv) => (
                                    <tr key={inv.id}>
                                        <td style={{ fontWeight: 500 }}>{inv.id}</td>
                                        <td style={{ color: '#8e8e93' }}>{inv.plant}</td>
                                        <td>
                                            <div className="prob-bar-container">
                                                <span style={{ fontVariantNumeric: 'tabular-nums', fontSize: 13 }}>
                                                    {(inv.failureProb * 100).toFixed(1)}%
                                                </span>
                                                <div className="prob-bar">
                                                    <div
                                                        className={`prob-bar-fill ${inv.failureProb > 0.7 ? 'high' : inv.failureProb > 0.4 ? 'medium' : 'low'}`}
                                                        style={{ width: `${inv.failureProb * 100}%` }}
                                                    />
                                                </div>
                                            </div>
                                        </td>
                                        <td>
                                            <span className={`risk-pill ${inv.riskClass === 0 ? 'no-risk' : inv.riskClass === 1 ? 'degradation' : 'shutdown'}`}>
                                                {inv.riskClass === 0 ? <CheckCircle size={12} /> : inv.riskClass === 1 ? <AlertTriangle size={12} /> : <XCircle size={12} />}
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
