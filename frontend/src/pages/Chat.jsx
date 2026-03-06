import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Bot, User } from 'lucide-react'
import { postChat } from '../api'

const SUGGESTIONS = [
    'What drives inverter failures?',
    'Recommended maintenance actions',
    'Explain SHAP feature importance',
    'Summarize model accuracy',
]

export default function Chat() {
    const [msgs, setMsgs] = useState([
        { role: 'ai', text: "Hi — ask me anything about inverter health, risk factors, or maintenance planning.", src: '' }
    ])
    const [input, setInput] = useState('')
    const [busy, setBusy] = useState(false)
    const bottom = useRef(null)

    useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

    const send = async (text) => {
        const q = text || input.trim()
        if (!q) return
        setMsgs(p => [...p, { role: 'user', text: q }])
        setInput('')
        setBusy(true)
        try {
            const r = await postChat(q)
            setMsgs(p => [...p, { role: 'ai', text: r.response || 'Something went wrong.', src: r.source || '' }])
        } catch {
            setMsgs(p => [...p, { role: 'ai', text: 'Cannot reach the API. Is the backend running on port 8000?', src: 'error' }])
        }
        setBusy(false)
    }

    return (
        <>
            <motion.div className="page-header" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <h2>AI Copilot</h2>
                <p>Powered by Gemini — ask about inverter diagnostics</p>
            </motion.div>

            <div className="card rise">
                <div className="chat-wrap">
                    <div className="chat-feed">
                        {msgs.map((m, i) => (
                            <div key={i} className={`msg ${m.role === 'user' ? 'from-user' : ''}`}>
                                <div className={`msg-avatar ${m.role === 'ai' ? 'bot' : 'human'}`}>
                                    {m.role === 'ai' ? <Bot size={13} /> : <User size={13} />}
                                </div>
                                <div className="msg-body">
                                    {m.text}
                                    {m.src && m.src !== 'error' && m.src !== '' && (
                                        <span className="src">via {m.src}</span>
                                    )}
                                </div>
                            </div>
                        ))}
                        {busy && (
                            <div className="msg">
                                <div className="msg-avatar bot"><Bot size={13} /></div>
                                <div className="msg-body" style={{ color: '#c4c4c4' }}>Thinking…</div>
                            </div>
                        )}
                        <div ref={bottom} />
                    </div>

                    <div className="chat-bottom">
                        <div className="chat-row">
                            <input
                                className="chat-input"
                                placeholder="Ask a question…"
                                value={input}
                                onChange={e => setInput(e.target.value)}
                                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), send())}
                                disabled={busy}
                            />
                            <button className="chat-send" onClick={() => send()} disabled={busy}>
                                <Send size={15} />
                            </button>
                        </div>
                        <div className="suggestions">
                            {SUGGESTIONS.map((s, i) => (
                                <button key={i} className="sug-btn" onClick={() => send(s)}>{s}</button>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
