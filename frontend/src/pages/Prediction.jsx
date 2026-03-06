import { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Zap, AlertTriangle, CheckCircle, XCircle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { postPredict } from '../api'

const fadeUp = {
    initial: { opacity: 0, y: 16 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] }
}

const tooltipStyle = {
    background: '#fff', border: '1px solid #e8e5e1', borderRadius: 10,
    fontSize: 12, color: '#1a1a1a', boxShadow: '0 4px 12px rgba(0,0,0,0.06)'
}

const RISK_LABELS = { 0: 'No Risk', 1: 'Degradation Risk', 2: 'Shutdown Risk' }
const RISK_ICONS = { 0: CheckCircle, 1: AlertTriangle, 2: XCircle }
const RISK_COLORS = { 0: '#3eb489', 1: '#d4a843', 2: '#d45f58' }

export default function Prediction() {
    const [form, setForm] = useState({
        inv_power: 5000, inv_temp: 45, meter_freq: 50.0,
        alarm_count_7d: 0, meter_pf: 0.98, hour: 12,
    })
    const [result, setResult] = useState(null)
    const [history, setHistory] = useState([])
    const [loading, setLoading] = useState(false)

    const handleChange = (key, value) => setForm(prev => ({ ...prev, [key]: parseFloat(value) || 0 }))

    const handlePredict = async () => {
        setLoading(true)
        try {
            const features = {
                ...form,
                is_daytime: form.hour >= 6 && form.hour <= 18 ? 1 : 0,
                freq_deviation: Math.abs(form.meter_freq - 50.0),
                pf_deviation: Math.abs(1.0 - form.meter_pf),
            }
            const res = await postPredict(features)
            setResult(res)
            setHistory(prev => [...prev, { time: new Date().toLocaleTimeString(), prob: res.failure_probability }])
        } catch {
            setResult({ error: true })
        }
        setLoading(false)
    }

    const RiskIcon = result && !result.error ? RISK_ICONS[result.risk_class] || AlertTriangle : null

    const fields = [
        { key: 'inv_power', label: 'Power Output (W)', step: undefined },
        { key: 'inv_temp', label: 'Temperature (°C)', step: undefined },
        { key: 'meter_freq', label: 'Grid Frequency (Hz)', step: '0.1' },
        { key: 'alarm_count_7d', label: 'Alarm Count (7d)', step: undefined },
        { key: 'meter_pf', label: 'Power Factor', step: '0.01' },
        { key: 'hour', label: 'Hour of Day (0–23)', step: undefined },
    ]

    return (
        <>
            <motion.div className="page-header" {...fadeUp}>
                <h2>Live Prediction</h2>
                <p>Enter inverter telemetry for real-time failure risk assessment</p>
            </motion.div>

            <div className="bento-grid">
                <motion.div className="bento-card span-5" {...fadeUp} transition={{ delay: 0.05 }}>
                    <div className="card-header">
                        <div className="card-title"><Activity size={15} /> Inverter Parameters</div>
                    </div>
                    <div className="prediction-controls">
                        {fields.map(({ key, label, step }) => (
                            <div className="form-group" key={key}>
                                <label>{label}</label>
                                <input
                                    type="number"
                                    step={step}
                                    value={form[key]}
                                    onChange={e => handleChange(key, e.target.value)}
                                />
                            </div>
                        ))}
                        <button className="predict-btn" onClick={handlePredict} disabled={loading}>
                            {loading ? 'Analyzing…' : 'Predict Risk'}
                        </button>
                    </div>
                </motion.div>

                <motion.div className="bento-card span-7" {...fadeUp} transition={{ delay: 0.1 }}>
                    <div className="card-header">
                        <div className="card-title"><Zap size={15} /> Prediction Result</div>
                        {result && !result.error && (
                            <span className={`card-badge ${result.risk_class === 0 ? 'badge-green' : result.risk_class === 1 ? 'badge-coral' : 'badge-red'}`}>
                                {RISK_LABELS[result.risk_class]}
                            </span>
                        )}
                    </div>

                    {!result ? (
                        <div style={{ textAlign: 'center', padding: '48px 0', color: '#bfbfbf' }}>
                            <Activity size={36} style={{ marginBottom: 10, opacity: 0.4 }} />
                            <p style={{ fontSize: 14 }}>Enter parameters and click Predict</p>
                        </div>
                    ) : result.error ? (
                        <div style={{ textAlign: 'center', padding: '48px 0', color: '#d45f58' }}>
                            <p>Could not reach the API. Is the backend running?</p>
                        </div>
                    ) : (
                        <>
                            <div className="prediction-result">
                                <div className="result-card">
                                    <div className="result-value" style={{ color: result.failure_probability > 0.5 ? '#d45f58' : '#3eb489' }}>
                                        {(result.failure_probability * 100).toFixed(1)}%
                                    </div>
                                    <div className="result-label">Failure Probability</div>
                                </div>
                                <div className="result-card">
                                    <div className="result-value" style={{ color: RISK_COLORS[result.risk_class], fontSize: 18 }}>
                                        {RiskIcon && <RiskIcon size={22} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 5 }} />}
                                        {RISK_LABELS[result.risk_class]}
                                    </div>
                                    <div className="result-label">Risk Level</div>
                                </div>
                                <div className="result-card">
                                    <div className="result-value" style={{ color: result.is_anomaly ? '#d45f58' : '#3eb489', fontSize: 20 }}>
                                        {result.is_anomaly ? 'Yes' : 'No'}
                                    </div>
                                    <div className="result-label">Anomaly Detected</div>
                                </div>
                            </div>

                            {result.top_risk_factors?.length > 0 && (
                                <div style={{ marginTop: 22 }}>
                                    <div className="card-title" style={{ marginBottom: 10, fontSize: 13 }}>
                                        <Zap size={14} /> Key Risk Drivers
                                    </div>
                                    <div className="shap-list">
                                        {result.top_risk_factors.map((f, i) => (
                                            <div className="shap-item" key={i}>
                                                <div className="shap-rank">{i + 1}</div>
                                                <div className="shap-name">{f.feature}</div>
                                                <span style={{
                                                    fontSize: 11, padding: '2px 8px', borderRadius: 100,
                                                    background: f.direction === 'increases risk' ? 'rgba(212,95,88,0.08)' : 'rgba(62,180,137,0.08)',
                                                    color: f.direction === 'increases risk' ? '#d45f58' : '#3eb489',
                                                }}>
                                                    {f.direction === 'increases risk' ? '↑ Risk' : '↓ Risk'}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <div style={{
                                marginTop: 18, padding: '12px 16px', borderRadius: 12,
                                background: result.risk_class === 0 ? 'rgba(62,180,137,0.05)' :
                                    result.risk_class === 1 ? 'rgba(212,168,67,0.05)' : 'rgba(212,95,88,0.05)',
                                border: `1px solid ${result.risk_class === 0 ? 'rgba(62,180,137,0.12)' :
                                    result.risk_class === 1 ? 'rgba(212,168,67,0.12)' : 'rgba(212,95,88,0.12)'}`,
                                fontSize: 13, lineHeight: 1.6, color: '#6b6b6b'
                            }}>
                                {result.recommendation}
                            </div>
                        </>
                    )}
                </motion.div>

                {history.length > 1 && (
                    <motion.div className="bento-card span-12" {...fadeUp}>
                        <div className="card-header">
                            <div className="card-title"><Activity size={15} /> Prediction History</div>
                            <span style={{ fontSize: 11, color: '#999' }}>{history.length} predictions</span>
                        </div>
                        <div className="chart-container" style={{ height: 160 }}>
                            <ResponsiveContainer>
                                <LineChart data={history}>
                                    <XAxis dataKey="time" tick={{ fill: '#999', fontSize: 11 }} axisLine={false} tickLine={false} />
                                    <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                                        tick={{ fill: '#999', fontSize: 11 }} axisLine={false} tickLine={false} />
                                    <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} contentStyle={tooltipStyle} />
                                    <Line type="monotone" dataKey="prob" stroke="#e8736c" strokeWidth={2} dot={{ fill: '#e8736c', r: 3.5 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </motion.div>
                )}
            </div>
        </>
    )
}
