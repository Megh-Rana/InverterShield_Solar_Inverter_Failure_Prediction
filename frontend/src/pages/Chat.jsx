import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Bot, User, Sparkles } from 'lucide-react'
import { postChat } from '../api'

const fadeUp = {
    initial: { opacity: 0, y: 16 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] }
}

const QUICK_QUESTIONS = [
    'What are the most important failure indicators?',
    'What maintenance do you recommend?',
    'How does SHAP explain predictions?',
    'What are the common root causes for inverter failures?',
    'Summarize the model performance',
]

export default function Chat() {
    const [messages, setMessages] = useState([
        {
            role: 'ai',
            content: "Hi there — I'm the InverterShield copilot. I can help you understand failure predictions, identify risk drivers, and recommend maintenance actions. What would you like to know?",
            source: 'system',
        }
    ])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const messagesEndRef = useRef(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const sendMessage = async (text) => {
        const msg = text || input.trim()
        if (!msg) return

        setMessages(prev => [...prev, { role: 'user', content: msg }])
        setInput('')
        setLoading(true)

        try {
            const res = await postChat(msg)
            setMessages(prev => [...prev, {
                role: 'ai',
                content: res.response || 'I had trouble with that. Try again?',
                source: res.source || 'api',
            }])
        } catch {
            setMessages(prev => [...prev, {
                role: 'ai',
                content: 'Could not connect to the backend. Make sure the API is running on port 8000.',
                source: 'error',
            }])
        }

        setLoading(false)
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    return (
        <>
            <motion.div className="page-header" {...fadeUp}>
                <h2>AI Copilot</h2>
                <p>Ask about inverter health, SHAP analysis, or maintenance planning</p>
            </motion.div>

            <motion.div className="bento-card" {...fadeUp} transition={{ delay: 0.05 }}
                style={{ gridColumn: 'span 12' }}>
                <div className="chat-container">
                    <div className="chat-messages">
                        {messages.map((msg, i) => (
                            <motion.div
                                key={i}
                                className={`chat-message ${msg.role}`}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.25 }}
                            >
                                <div className={`chat-avatar ${msg.role}`}>
                                    {msg.role === 'ai' ? <Bot size={14} /> : <User size={14} />}
                                </div>
                                <div className="chat-bubble">
                                    <div style={{ whiteSpace: 'pre-wrap' }}>
                                        {msg.content.split('**').map((part, j) =>
                                            j % 2 === 1 ? <strong key={j}>{part}</strong> : part
                                        )}
                                    </div>
                                    {msg.source && msg.role === 'ai' && msg.source !== 'system' && (
                                        <div className="source-tag">
                                            <Sparkles size={9} style={{ display: 'inline', marginRight: 3, verticalAlign: 'middle' }} />
                                            {msg.source}
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        ))}

                        {loading && (
                            <motion.div className="chat-message ai" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                <div className="chat-avatar ai"><Bot size={14} /></div>
                                <div className="chat-bubble" style={{ color: '#bfbfbf' }}>Thinking…</div>
                            </motion.div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    <div className="chat-input-container">
                        <input
                            className="chat-input"
                            placeholder="Ask about inverter health, SHAP features, maintenance…"
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            disabled={loading}
                        />
                        <button className="chat-send-btn" onClick={() => sendMessage()} disabled={loading}>
                            <Send size={16} />
                        </button>
                    </div>

                    <div className="quick-actions">
                        {QUICK_QUESTIONS.map((q, i) => (
                            <button key={i} className="quick-action-btn" onClick={() => sendMessage(q)}>
                                {q}
                            </button>
                        ))}
                    </div>
                </div>
            </motion.div>
        </>
    )
}
