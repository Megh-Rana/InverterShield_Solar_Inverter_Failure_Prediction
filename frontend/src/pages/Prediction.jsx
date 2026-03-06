import { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Thermometer, Zap, Radio, AlertTriangle, CheckCircle, XCircle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { postPredict } from '../api'

const fadeUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] }
}

const RISK_LABELS = { 0: 'No Risk', 1: 'Degradation Risk', 2: 'Shutdown Risk' }
const RISK_ICONS = { 0: CheckCircle, 1: AlertTriangle, 2: XCircle }
const RISK_COLORS = { 0: '#34d399', 1: '#fbbf24', 2: '#ff4757' }

export default function Prediction() {
    const [form, setForm] = useState({
        inv_power: 5000,
        inv_temp: 45,
        meter_freq: 50.0,
        alarm_count_7d: 0,
        meter_pf: 0.98,
        hour: 12,
    })
    const [result, setResult] = useState(null)
    const [history, setHistory] = useState([])
    const [loading, setLoading] = useState(false)

    const handleChange = (key, value) => {
        setForm(prev => ({ ...prev, [key]: parseFloat(value) || 0 }))
    }

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
            setHistory(prev => [...prev, {
                time: new Date().toLocaleTimeString(),
                prob: res.failure_probability,
                risk: res.risk_class,
            }])
        } catch {
            setResult({ error: true })
        }
        setLoading(false)
    }

    const RiskIcon = result ? RISK_ICONS[result.risk_class] || AlertTriangle : null

    return (
        <>
            <motion.div className="page-header" {...fadeUp}>
                <h2>Live Prediction</h2>
                <p>Enter inverter telemetry data for real-time failure risk assessment</p>
            </motion.div>

            <div className="bento-grid">
                {/* Input Form */}
                <motion.div className="bento-card span-5" {...fadeUp} transition={{ delay: 0.05 }}>
                    <div className="card-header">
                        <div className="card-title"><Activity size={16} /> Inverter Parameters</div>
                    </div>
                    <div className="prediction-controls">
                        <div className="form-group">
                            <label>Power Output (W)</label>
                            <input type="number" value={form.inv_power} onChange={e => handleChange('inv_power', e.target.value)} />
                        </div>
                        <div className="form-group">
                            <label>Temperature (°C)</label>
                            <input type="number" value={form.inv_temp} onChange={e => handleChange('inv_temp', e.target.value)} />
                        </div>
                        <div className="form-group">
                            <label>Grid Frequency (Hz)</label>
                            <input type="number" step="0.1" value={form.meter_freq} onChange={e => handleChange('meter_freq', e.target.value)} />
                        </div>
                        <div className="form-group">
                            <label>Alarm Count (7d)</label>
                            <input type="number" value={form.alarm_count_7d} onChange={e => handleChange('alarm_count_7d', e.target.value)} />
                        </div>
                        <div className="form-group">
                            <label>Power Factor</label>
                            <input type="number" step="0.01" value={form.meter_pf} onChange={e => handleChange('meter_pf', e.target.value)} />
                        </div>
                        <div className="form-group">
                            <label>Hour of Day</label>
                            <input type="number" min="0" max="23" value={form.hour} onChange={e => handleChange('hour', e.target.value)} />
                        </div>
                        <button className="predict-btn" onClick={handlePredict} disabled={loading}>
                            {loading ? 'Analyzing...' : '⚡ Predict Risk'}
                        </button>
                    </div>
                </motion.div>

                {/* Result */}
                <motion.div className="bento-card span-7" {...fadeUp} transition={{ delay: 0.1 }}>
                    <div className="card-header">
                        <div className="card-title"><Zap size={16} /> Prediction Result</div>
                        {result && !result.error && (
                            <span className={`card-badge ${result.risk_class === 0 ? 'badge-green' : result.risk_class === 1 ? 'badge-orange' : 'badge-red'}`}>
                                {RISK_LABELS[result.risk_class]}
                            </span>
                        )}
                    </div>

                    {!result ? (
                        <div style={{ textAlign: 'center', padding: '40px 0', color: '#636366' }}>
                            <Activity size={40} style={{ marginBottom: 12, opacity: 0.3 }} />
                            <p>Enter parameters and click Predict to see results</p>
                        </div>
                    ) : result.error ? (
                        <div style={{ textAlign: 'center', padding: '40px 0', color: '#ff4757' }}>
                            <p>Failed to connect to API. Is the backend running?</p>
                        </div>
                    ) : (
                        <>
                            <div className="prediction-result">
                                <div className="result-card">
                                    <div className="result-value" style={{ color: result.failure_probability > 0.5 ? '#ff4757' : '#34d399' }}>
                                        {(result.failure_probability * 100).toFixed(1)}%
                                    </div>
                                    <div className="result-label">Failure Probability</div>
                                </div>
                                <div className="result-card">
                                    <div className="result-value" style={{ color: RISK_COLORS[result.risk_class], fontSize: 20 }}>
                                        {RiskIcon && <RiskIcon size={28} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />}
                                        {RISK_LABELS[result.risk_class]}
                                    </div>
                                    <div className="result-label">Risk Level</div>
                                </div>
                                <div className="result-card">
                                    <div className="result-value" style={{ color: result.is_anomaly ? '#ff4757' : '#34d399', fontSize: 22 }}>
                                        {result.is_anomaly ? '⚠️ Yes' : '✅ No'}
                                    </div>
                                    <div className="result-label">Anomaly Detected</div>
                                </div>
                            </div>

                            {/* SHAP Factors */}
                            {result.top_risk_factors && result.top_risk_factors.length > 0 && (
                                <div style={{ marginTop: 24 }}>
                                    <div className="card-title" style={{ marginBottom: 12, fontSize: 13 }}>
                                        <Zap size={14} /> Key Risk Drivers
                                    </div>
                                    <div className="shap-list">
                                        {result.top_risk_factors.map((f, i) => (
                                            <div className="shap-item" key={i}>
                                                <div className="shap-rank">{i + 1}</div>
                                                <div className="shap-name">{f.feature}</div>
                                                <span style={{
                                                    fontSize: 11,
                                                    padding: '2px 8px',
                                                    borderRadius: 100,
                                                    background: f.direction === 'increases risk' ? 'rgba(255,71,87,0.1)' : 'rgba(52,211,153,0.1)',
                                                    color: f.direction === 'increases risk' ? '#ff4757' : '#34d399',
                                                }}>
                                                    {f.direction === 'increases risk' ? '↑ Risk' : '↓ Risk'}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <div style={{
                                marginTop: 20, padding: '14px 16px', borderRadius: 12,
                                background: result.risk_class === 0 ? 'rgba(52,211,153,0.06)' :
                                    result.risk_class === 1 ? 'rgba(251,191,36,0.06)' : 'rgba(255,71,87,0.06)',
                                border: `1px solid ${result.risk_class === 0 ? 'rgba(52,211,153,0.15)' :
                                    result.risk_class === 1 ? 'rgba(251,191,36,0.15)' : 'rgba(255,71,87,0.15)'}`,
                                fontSize: 13, lineHeight: 1.6, color: '#8e8e93'
                            }}>
                                {result.recommendation}
                            </div>
                        </>
                    )}
                </motion.div>

                {/* History Chart */}
                {history.length > 1 && (
                    <motion.div className="bento-card span-12" {...fadeUp}>
                        <div className="card-header">
                            <div className="card-title"><Activity size={16} /> Prediction History</div>
                            <span style={{ fontSize: 11, color: '#636366' }}>{history.length} predictions</span>
                        </div>
                        <div className="chart-container" style={{ height: 180 }}>
                            <ResponsiveContainer>
                                <LineChart data={history}>
                                    <XAxis dataKey="time" tick={{ fill: '#636366', fontSize: 11 }} axisLine={false} tickLine={false} />
                                    <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                                        tick={{ fill: '#636366', fontSize: 11 }} axisLine={false} tickLine={false} />
                                    <Tooltip
                                        formatter={v => `${(v * 100).toFixed(1)}%`}
                                        contentStyle={{ background: '#0f0f15', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12, color: '#f5f5f7' }}
                                    />
                                    <Line type="monotone" dataKey="prob" stroke="#ff6b2b" strokeWidth={2} dot={{ fill: '#ff6b2b', r: 4 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </motion.div>
                )}
            </div>
        </>
    )
}
