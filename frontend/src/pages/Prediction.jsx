import { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { postPredict } from '../api'

const tip = {
    background: '#fff', border: '1px solid #eae8e4', borderRadius: 8,
    fontSize: 11, color: '#202020', boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
}

const LABELS = { 0: 'No Risk', 1: 'Degradation', 2: 'Shutdown Risk' }
const CLRS = { 0: '#3ea87d', 1: '#c09a3a', 2: '#d9635e' }

export default function Prediction() {
    const [form, setForm] = useState({
        inv_power: 5000, inv_temp: 45, meter_freq: 50.0,
        alarm_count_7d: 0, meter_pf: 0.98, hour: 12,
    })
    const [result, setResult] = useState(null)
    const [history, setHistory] = useState([])
    const [loading, setLoading] = useState(false)

    const set = (k, v) => setForm(p => ({ ...p, [k]: parseFloat(v) || 0 }))

    const predict = async () => {
        setLoading(true)
        try {
            const feats = {
                ...form,
                is_daytime: form.hour >= 6 && form.hour <= 18 ? 1 : 0,
                freq_deviation: Math.abs(form.meter_freq - 50.0),
                pf_deviation: Math.abs(1.0 - form.meter_pf),
            }
            const r = await postPredict(feats)
            setResult(r)
            setHistory(p => [...p, { t: new Date().toLocaleTimeString(), prob: r.failure_probability }])
        } catch { setResult({ error: true }) }
        setLoading(false)
    }

    const fields = [
        ['inv_power', 'Power (W)'],
        ['inv_temp', 'Temperature (°C)'],
        ['meter_freq', 'Grid freq (Hz)', '0.1'],
        ['alarm_count_7d', 'Alarms (7d)'],
        ['meter_pf', 'Power factor', '0.01'],
        ['hour', 'Hour (0–23)'],
    ]

    return (
        <>
            <motion.div className="page-header" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <h2>Live Prediction</h2>
                <p>Enter telemetry to assess failure risk</p>
            </motion.div>

            <div className="grid-5-7">
                <div className="card rise">
                    <div className="card-head"><h3>Parameters</h3></div>
                    <div className="param-grid">
                        {fields.map(([k, label, step]) => (
                            <div className="param-group" key={k}>
                                <label>{label}</label>
                                <input type="number" step={step} value={form[k]} onChange={e => set(k, e.target.value)} />
                            </div>
                        ))}
                        <button className="run-btn" onClick={predict} disabled={loading}>
                            {loading ? 'Running…' : 'Run prediction'}
                        </button>
                    </div>
                </div>

                <div className="card rise">
                    <div className="card-head">
                        <h3>Result</h3>
                        {result && !result.error && (
                            <span className="tag" style={{ color: CLRS[result.risk_class] }}>{LABELS[result.risk_class]}</span>
                        )}
                    </div>

                    {!result ? (
                        <div style={{ textAlign: 'center', padding: '40px 0', color: '#c4c4c4' }}>
                            <Activity size={28} style={{ marginBottom: 8, opacity: 0.4 }} />
                            <p style={{ fontSize: 13 }}>Waiting for input</p>
                        </div>
                    ) : result.error ? (
                        <p style={{ padding: '40px 0', textAlign: 'center', color: '#d9635e', fontSize: 13 }}>
                            Cannot reach API — is the backend running?
                        </p>
                    ) : (
                        <>
                            <div className="result-row">
                                <div className="result-box">
                                    <div className="val" style={{ color: result.failure_probability > 0.5 ? '#d9635e' : '#3ea87d' }}>
                                        {(result.failure_probability * 100).toFixed(1)}%
                                    </div>
                                    <div className="lbl">Failure probability</div>
                                </div>
                                <div className="result-box">
                                    <div className="val" style={{ color: CLRS[result.risk_class], fontSize: 17 }}>
                                        {LABELS[result.risk_class]}
                                    </div>
                                    <div className="lbl">Risk level</div>
                                </div>
                                <div className="result-box">
                                    <div className="val" style={{ color: result.is_anomaly ? '#d9635e' : '#3ea87d', fontSize: 18 }}>
                                        {result.is_anomaly ? 'Yes' : 'No'}
                                    </div>
                                    <div className="lbl">Anomaly</div>
                                </div>
                            </div>

                            {result.top_risk_factors?.length > 0 && (
                                <div style={{ marginTop: 18 }}>
                                    <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Key drivers</h3>
                                    <div className="feature-list">
                                        {result.top_risk_factors.map((f, i) => (
                                            <div className="feature-row" key={i}>
                                                <span className="num">{i + 1}</span>
                                                <span className="fname">{f.feature}</span>
                                                <span style={{
                                                    fontSize: 10, padding: '1px 6px', borderRadius: 4,
                                                    background: f.direction === 'increases risk' ? 'rgba(217,99,94,0.07)' : 'rgba(62,168,125,0.07)',
                                                    color: f.direction === 'increases risk' ? '#d9635e' : '#3ea87d',
                                                }}>
                                                    {f.direction === 'increases risk' ? '↑' : '↓'}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {result.recommendation && (
                                <p style={{ marginTop: 14, padding: '10px 14px', background: 'var(--bg)', borderRadius: 8, fontSize: 12.5, color: '#555', lineHeight: 1.6 }}>
                                    {result.recommendation}
                                </p>
                            )}
                        </>
                    )}
                </div>
            </div>

            {history.length > 1 && (
                <div className="card rise" style={{ marginTop: 16 }}>
                    <div className="card-head">
                        <h3>History</h3>
                        <span style={{ fontSize: 11, color: '#999' }}>{history.length} runs</span>
                    </div>
                    <div className="chart-area" style={{ height: 150 }}>
                        <ResponsiveContainer>
                            <LineChart data={history}>
                                <XAxis dataKey="t" tick={{ fill: '#999', fontSize: 10 }} axisLine={false} tickLine={false} />
                                <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100) | 0}%`}
                                    tick={{ fill: '#999', fontSize: 10 }} axisLine={false} tickLine={false} />
                                <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} contentStyle={tip} />
                                <Line type="monotone" dataKey="prob" stroke="#d9635e" strokeWidth={1.5} dot={{ fill: '#d9635e', r: 3 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}
        </>
    )
}
