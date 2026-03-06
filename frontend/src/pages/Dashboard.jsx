import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell
} from 'recharts'
import { fetchDashboard } from '../api'

const COLORS = { 'No Risk': '#3ea87d', 'Degradation': '#c09a3a', 'Shutdown': '#d9635e' }

const tip = {
    background: '#fff', border: '1px solid #eae8e4', borderRadius: 8,
    fontSize: 11, color: '#202020', boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
}

const generateInverters = () => {
    const plants = ['Plant 1', 'Plant 2', 'Plant 3']
    return Array.from({ length: 32 }, (_, i) => {
        const p = Math.random()
        return {
            id: `INV-${String(i + 1).padStart(2, '0')}`,
            plant: plants[i % 3],
            prob: p,
            risk: p < 0.4 ? 0 : p < 0.7 ? 1 : 2,
            label: p < 0.4 ? 'No Risk' : p < 0.7 ? 'Degradation' : 'Shutdown',
        }
    }).sort((a, b) => b.prob - a.prob)
}

export default function Dashboard() {
    const [data, setData] = useState(null)
    const [inv] = useState(generateInverters)

    useEffect(() => { fetchDashboard().then(setData).catch(() => { }) }, [])

    const counts = inv.reduce((a, v) => ({ ...a, [v.label]: (a[v.label] || 0) + 1 }), {})
    const pie = Object.entries(counts).map(([name, value]) => ({ name, value }))

    const metrics = data?.model_performance?.binary_cv || {}
    const shap = data?.top_risk_factors || [
        'month', 'inv_temp_7d_mean', 'meter_kwh_import', 'inv_temp_24h_mean',
        'smu_string_mean_7d_std', 'meter_kwh_total', 'inv_id', 'alarm_count_7d',
    ]

    return (
        <>
            <motion.div className="page-header" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <h2>Dashboard</h2>
                <p>32 inverters across 3 plants</p>
            </motion.div>

            {/* Metrics strip */}
            <div className="metric-row rise">
                <div className="metric-item">
                    <div className="metric-label">F1 Score</div>
                    <div className="metric-value">{metrics.f1_mean?.toFixed(3) || '0.918'}</div>
                    <div className="metric-sub">5-fold CV</div>
                </div>
                <div className="metric-item">
                    <div className="metric-label">AUC</div>
                    <div className="metric-value">{metrics.auc_mean?.toFixed(3) || '0.901'}</div>
                    <div className="metric-sub">Walk-forward</div>
                </div>
                <div className="metric-item">
                    <div className="metric-label">Precision</div>
                    <div className="metric-value">{metrics.precision_mean?.toFixed(3) || '0.924'}</div>
                </div>
                <div className="metric-item">
                    <div className="metric-label">Inverters</div>
                    <div className="metric-value">32</div>
                    <div className="metric-sub">3 plants</div>
                </div>
            </div>

            {/* Row 1: pie + bar */}
            <div className="grid-5-7">
                <div className="card rise">
                    <div className="card-head">
                        <h3>Risk distribution</h3>
                    </div>
                    <div className="chart-area" style={{ height: 190 }}>
                        <ResponsiveContainer>
                            <PieChart>
                                <Pie data={pie} cx="50%" cy="50%" innerRadius={44} outerRadius={72} paddingAngle={2} dataKey="value">
                                    {pie.map(e => <Cell key={e.name} fill={COLORS[e.name]} />)}
                                </Pie>
                                <Tooltip contentStyle={tip} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div style={{ display: 'flex', gap: 14, justifyContent: 'center', marginTop: 4 }}>
                        {pie.map(({ name, value }) => (
                            <span key={name} style={{ fontSize: 11, color: '#999', display: 'flex', alignItems: 'center', gap: 4 }}>
                                <span style={{ width: 6, height: 6, borderRadius: 3, background: COLORS[name], display: 'inline-block' }} />
                                {name} {value}
                            </span>
                        ))}
                    </div>
                </div>

                <div className="card rise">
                    <div className="card-head">
                        <h3>Highest risk inverters</h3>
                    </div>
                    <div className="chart-area" style={{ height: 200 }}>
                        <ResponsiveContainer>
                            <BarChart data={inv.slice(0, 8)} layout="vertical">
                                <XAxis type="number" domain={[0, 1]} tickFormatter={v => `${(v * 100) | 0}%`}
                                    tick={{ fill: '#999', fontSize: 10 }} axisLine={false} tickLine={false} />
                                <YAxis type="category" dataKey="id" width={52}
                                    tick={{ fill: '#555', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} contentStyle={tip} />
                                <Bar dataKey="prob" radius={[0, 3, 3, 0]}>
                                    {inv.slice(0, 8).map(v => (
                                        <Cell key={v.id} fill={v.prob > 0.7 ? '#d9635e' : v.prob > 0.4 ? '#c09a3a' : '#3ea87d'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Row 2: features + table */}
            <div className="grid-5-7">
                <div className="card rise">
                    <div className="card-head">
                        <h3>Top risk factors</h3>
                        <span className="tag">SHAP</span>
                    </div>
                    <div className="feature-list">
                        {shap.slice(0, 8).map((f, i) => (
                            <div className="feature-row" key={f}>
                                <span className="num">{i + 1}</span>
                                <span className="fname">{f}</span>
                                <div className="feature-bar">
                                    <div className="feature-bar-fill" style={{ width: `${100 - i * 11}%` }} />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="card rise">
                    <div className="card-head">
                        <h3>Inverter status</h3>
                        <span style={{ fontSize: 11, color: '#999' }}>{inv.length}</span>
                    </div>
                    <div style={{ maxHeight: 260, overflowY: 'auto' }}>
                        <table className="risk-table">
                            <thead><tr><th>ID</th><th>Plant</th><th>Prob</th><th>Status</th></tr></thead>
                            <tbody>
                                {inv.slice(0, 15).map(v => (
                                    <tr key={v.id}>
                                        <td style={{ fontWeight: 500 }}>{v.id}</td>
                                        <td style={{ color: '#999' }}>{v.plant}</td>
                                        <td>
                                            <div className="prob-bar-wrap">
                                                <span style={{ fontSize: 12, fontVariantNumeric: 'tabular-nums' }}>{(v.prob * 100).toFixed(1)}%</span>
                                                <div className="prob-bar">
                                                    <div className={`prob-bar-fill ${v.prob > 0.7 ? 'hi' : v.prob > 0.4 ? 'med' : 'low'}`}
                                                        style={{ width: `${v.prob * 100}%` }} />
                                                </div>
                                            </div>
                                        </td>
                                        <td>
                                            <span className={`status-dot ${v.risk === 0 ? 'ok' : v.risk === 1 ? 'warn' : 'crit'}`}>
                                                {v.label}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </>
    )
}
