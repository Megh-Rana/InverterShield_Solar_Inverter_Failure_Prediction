import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, Zap, Thermometer, Radio, AlertTriangle, Clock, Sun, ChevronDown, ChevronUp, RotateCcw } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts'
import { postPredict } from '../api'

const tip = {
    background: '#fff', border: '1px solid #eae8e4', borderRadius: 8,
    fontSize: 11, color: '#202020', boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
}

const LABELS = { 0: 'No Risk', 1: 'Degradation', 2: 'Shutdown Risk' }
const CLRS = { 0: '#3ea87d', 1: '#c09a3a', 2: '#d9635e' }
const CLR_BG = { 0: 'rgba(62,168,125,0.08)', 1: 'rgba(192,154,58,0.08)', 2: 'rgba(217,99,94,0.08)' }

/* ── Telemetry input groups with labels, defaults, units ─────────────── */
const GROUPS = [
    {
        name: 'Inverter', icon: Zap, color: '#6366f1',
        fields: [
            ['inv_power', 'AC Power', 'W', 5000, 1],
            ['inv_temp', 'Temperature', '°C', 30, 0.5],
            ['inv_freq', 'Frequency', 'Hz', 50.0, 0.1],
            ['inv_v_ab', 'Voltage AB', 'V', 400, 1],
            ['inv_v_bc', 'Voltage BC', 'V', 400, 1],
            ['inv_v_ca', 'Voltage CA', 'V', 400, 1],
            ['inv_pv1_power', 'PV1 DC Power', 'W', 5000, 1],
            ['inv_kwh_today', 'Energy Today', 'kWh', 20000, 10],
        ]
    },
    {
        name: 'Grid / Meter', icon: Radio, color: '#0ea5e9',
        fields: [
            ['meter_pf', 'Power Factor', '', 0.98, 0.01],
            ['meter_freq', 'Grid Frequency', 'Hz', 50.0, 0.1],
            ['meter_v_r', 'Voltage R', 'V', 236, 0.5],
            ['meter_v_y', 'Voltage Y', 'V', 236, 0.5],
            ['meter_v_b', 'Voltage B', 'V', 236, 0.5],
            ['meter_meter_active_power', 'Active Power', 'kW', 10000, 10],
        ]
    },
    {
        name: 'Environment', icon: Sun, color: '#f59e0b',
        fields: [
            ['ambient_temp', 'Ambient Temp', '°C', 28, 0.5],
        ]
    },
    {
        name: 'String Monitoring', icon: Activity, color: '#8b5cf6',
        fields: [
            ['smu_string_mean', 'Mean Current', 'A', 5.0, 0.1],
            ['smu_string_std', 'Current Std Dev', 'A', 0.1, 0.01],
            ['smu_num_zero', 'Zero Strings', '', 0, 1],
            ['smu_total_strings', 'Total Strings', '', 24, 1],
        ]
    },
    {
        name: 'Alarms', icon: AlertTriangle, color: '#ef4444',
        fields: [
            ['alarm_count_24h', 'Alarms (24h)', '', 0, 1],
            ['alarm_count_7d', 'Alarms (7d)', '', 0, 1],
        ]
    },
    {
        name: 'Time', icon: Clock, color: '#64748b',
        fields: [
            ['hour', 'Hour (0–23)', '', 12, 1],
            ['month', 'Month (1–12)', '', new Date().getMonth() + 1, 1],
        ]
    },
]

/* ── Build initial form state from defaults ──────────────────────────── */
function buildDefaults() {
    const d = {}
    GROUPS.forEach(g => g.fields.forEach(([k, , , v]) => { d[k] = v }))
    return d
}

export default function Prediction() {
    const [form, setForm] = useState(buildDefaults)
    const [result, setResult] = useState(null)
    const [history, setHistory] = useState([])
    const [loading, setLoading] = useState(false)
    const [expanded, setExpanded] = useState({ 'Inverter': true, 'Grid / Meter': true, 'Alarms': true })

    const set = (k, v) => setForm(p => ({ ...p, [k]: parseFloat(v) || 0 }))
    const resetForm = () => { setForm(buildDefaults()); setResult(null) }

    const toggle = name => setExpanded(p => ({ ...p, [name]: !p[name] }))

    const predict = async () => {
        setLoading(true)
        try {
            const r = await postPredict(form)
            setResult(r)
            setHistory(p => [...p.slice(-19), { t: new Date().toLocaleTimeString(), prob: r.failure_probability, cls: r.risk_class }])
        } catch { setResult({ error: true }) }
        setLoading(false)
    }

    const shapData = result?.top_risk_factors?.map((f, i) => ({
        name: f.feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        raw: f.feature,
        impact: Math.abs(f.impact),
        dir: f.impact > 0 ? 1 : -1,
        value: f.value,
    })) || []

    return (
        <>
            <motion.div className="page-header" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <h2>Live Prediction</h2>
                <p>Provide inverter telemetry data — the ML model predicts failure risk in real-time</p>
            </motion.div>

            <div className="pred-layout">
                {/* ── LEFT: Telemetry Inputs ── */}
                <div className="pred-inputs">
                    <div className="card rise" style={{ overflow: 'visible' }}>
                        <div className="card-head" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <h3>Telemetry Parameters</h3>
                            <button className="reset-btn" onClick={resetForm} title="Reset to healthy defaults">
                                <RotateCcw size={14} /> Reset
                            </button>
                        </div>

                        {GROUPS.map(({ name, icon: Icon, color, fields }) => (
                            <div className="param-section" key={name}>
                                <div className="section-head" onClick={() => toggle(name)}>
                                    <Icon size={15} style={{ color }} />
                                    <span>{name}</span>
                                    <span className="field-count">{fields.length}</span>
                                    {expanded[name] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                </div>
                                <AnimatePresence>
                                    {expanded[name] && (
                                        <motion.div
                                            className="param-grid-new"
                                            initial={{ height: 0, opacity: 0 }}
                                            animate={{ height: 'auto', opacity: 1 }}
                                            exit={{ height: 0, opacity: 0 }}
                                            transition={{ duration: 0.2 }}
                                        >
                                            {fields.map(([k, label, unit, , step]) => (
                                                <div className="param-item" key={k}>
                                                    <label>{label}{unit && <span className="unit">{unit}</span>}</label>
                                                    <input
                                                        type="number"
                                                        step={step}
                                                        value={form[k]}
                                                        onChange={e => set(k, e.target.value)}
                                                    />
                                                </div>
                                            ))}
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        ))}

                        <button className="run-btn" onClick={predict} disabled={loading}>
                            {loading ? 'Running ML Model…' : '⚡ Run Prediction'}
                        </button>
                    </div>
                </div>

                {/* ── RIGHT: Results Panel ── */}
                <div className="pred-results">
                    <div className="card rise">
                        <div className="card-head">
                            <h3>Prediction Result</h3>
                            {result && !result.error && (
                                <span className="tag" style={{ color: CLRS[result.risk_class], background: CLR_BG[result.risk_class] }}>
                                    {LABELS[result.risk_class]}
                                </span>
                            )}
                        </div>

                        {!result ? (
                            <div style={{ textAlign: 'center', padding: '60px 0', color: '#c4c4c4' }}>
                                <Activity size={32} style={{ marginBottom: 12, opacity: 0.3 }} />
                                <p style={{ fontSize: 13 }}>Provide telemetry data and run a prediction</p>
                            </div>
                        ) : result.error ? (
                            <p style={{ padding: '60px 0', textAlign: 'center', color: '#d9635e', fontSize: 13 }}>
                                Cannot reach API — is the backend running?
                            </p>
                        ) : (
                            <>
                                {/* Risk gauge */}
                                <div className="result-row">
                                    <div className="result-box">
                                        <div className="val" style={{ color: result.failure_probability > 0.5 ? '#d9635e' : '#3ea87d', fontSize: 28 }}>
                                            {(result.failure_probability * 100).toFixed(1)}%
                                        </div>
                                        <div className="lbl">Failure Probability</div>
                                    </div>
                                    <div className="result-box">
                                        <div className="val" style={{ color: CLRS[result.risk_class], fontSize: 18, fontWeight: 600 }}>
                                            {LABELS[result.risk_class]}
                                        </div>
                                        <div className="lbl">Risk Classification</div>
                                    </div>
                                    <div className="result-box">
                                        <div className="val" style={{ color: result.is_anomaly ? '#d9635e' : '#3ea87d', fontSize: 18 }}>
                                            {result.is_anomaly ? '⚠ Yes' : '✓ No'}
                                        </div>
                                        <div className="lbl">Anomaly Detected</div>
                                    </div>
                                </div>

                                {/* SHAP feature importance chart */}
                                {shapData.length > 0 && (
                                    <div style={{ marginTop: 20 }}>
                                        <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 12 }}>
                                            SHAP Feature Importance (ML Explainability)
                                        </h3>
                                        <div style={{ height: 180 }}>
                                            <ResponsiveContainer>
                                                <BarChart data={shapData} layout="vertical" margin={{ left: 100, right: 20 }}>
                                                    <XAxis type="number" tick={{ fontSize: 10, fill: '#999' }} axisLine={false} tickLine={false} />
                                                    <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: '#666' }} axisLine={false} tickLine={false} width={95} />
                                                    <Tooltip
                                                        contentStyle={tip}
                                                        formatter={(v, n, props) => {
                                                            const item = props.payload
                                                            return [`SHAP impact: ${item.dir > 0 ? '+' : ''}${(item.dir * item.impact).toFixed(4)}`, `Value: ${item.value}`]
                                                        }}
                                                    />
                                                    <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                                                        {shapData.map((d, i) => (
                                                            <Cell key={i} fill={d.dir > 0 ? 'rgba(217,99,94,0.75)' : 'rgba(62,168,125,0.75)'} />
                                                        ))}
                                                    </Bar>
                                                </BarChart>
                                            </ResponsiveContainer>
                                        </div>
                                        <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginTop: 6 }}>
                                            <span style={{ fontSize: 10, color: '#d9635e' }}>■ Increases risk</span>
                                            <span style={{ fontSize: 10, color: '#3ea87d' }}>■ Decreases risk</span>
                                        </div>
                                    </div>
                                )}

                                {/* Recommendation */}
                                {result.recommendation && (
                                    <div className="recommendation-box" style={{
                                        background: CLR_BG[result.risk_class],
                                        borderLeft: `3px solid ${CLRS[result.risk_class]}`
                                    }}>
                                        {result.recommendation}
                                    </div>
                                )}
                            </>
                        )}
                    </div>

                    {/* Prediction history chart */}
                    {history.length > 1 && (
                        <div className="card rise" style={{ marginTop: 16 }}>
                            <div className="card-head">
                                <h3>Prediction History</h3>
                                <span style={{ fontSize: 11, color: '#999' }}>{history.length} runs</span>
                            </div>
                            <div className="chart-area" style={{ height: 140 }}>
                                <ResponsiveContainer>
                                    <LineChart data={history}>
                                        <XAxis dataKey="t" tick={{ fill: '#999', fontSize: 9 }} axisLine={false} tickLine={false} />
                                        <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100) | 0}%`}
                                            tick={{ fill: '#999', fontSize: 10 }} axisLine={false} tickLine={false} />
                                        <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} contentStyle={tip} />
                                        <Line type="monotone" dataKey="prob" stroke="#d9635e" strokeWidth={1.5} dot={{ fill: '#d9635e', r: 3 }} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </>
    )
}
