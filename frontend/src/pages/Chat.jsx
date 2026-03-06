import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { MessageSquare, Send, Bot, User, Sparkles } from 'lucide-react'
import { postChat } from '../api'

const fadeUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] }
}

const QUICK_QUESTIONS = [
    '📊 What is the model performance?',
    '🔍 What are the most important features?',
    '🔧 What maintenance actions do you recommend?',
    '⚡ Why do inverters fail?',
    '📈 How does SHAP explain predictions?',
]

export default function Chat() {
    const [messages, setMessages] = useState([
        {
            role: 'ai',
            content: "Hello! I'm the **InverterShield AI Copilot**. I can help you with:\n\n• Explaining failure predictions and risk levels\n• Identifying root causes from SHAP analysis\n• Recommending maintenance actions\n• Analyzing inverter performance trends\n\nWhat would you like to know?",
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
                content: res.response || 'I had trouble processing that. Please try again.',
                source: res.source || 'api',
            }])
        } catch {
            setMessages(prev => [...prev, {
                role: 'ai',
                content: 'Unable to connect to the backend. Make sure the API server is running on port 8000.',
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
                <h2>AI Operations Copilot</h2>
                <p>Ask about inverter health, risk factors, or maintenance — powered by Gemini AI</p>
            </motion.div>

            <motion.div className="bento-card span-12" {...fadeUp} transition={{ delay: 0.05 }}
                style={{ gridColumn: 'span 12' }}>
                <div className="chat-container">
                    {/* Messages */}
                    <div className="chat-messages">
                        {messages.map((msg, i) => (
                            <motion.div
                                key={i}
                                className={`chat-message ${msg.role}`}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.3 }}
                            >
                                <div className={`chat-avatar ${msg.role}`}>
                                    {msg.role === 'ai' ? <Bot size={16} /> : <User size={16} />}
                                </div>
                                <div className="chat-bubble">
                                    <div style={{ whiteSpace: 'pre-wrap' }}>
                                        {msg.content.split('**').map((part, j) =>
                                            j % 2 === 1 ? <strong key={j}>{part}</strong> : part
                                        )}
                                    </div>
                                    {msg.source && msg.role === 'ai' && msg.source !== 'system' && (
                                        <div className="source-tag">
                                            <Sparkles size={10} style={{ display: 'inline', marginRight: 3, verticalAlign: 'middle' }} />
                                            {msg.source}
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        ))}

                        {loading && (
                            <motion.div className="chat-message ai"
                                initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                <div className="chat-avatar ai"><Bot size={16} /></div>
                                <div className="chat-bubble" style={{ color: '#636366' }}>
                                    <span className="typing-indicator">Thinking</span>
                                    <span style={{ animation: 'pulse 1s infinite' }}>...</span>
                                </div>
                            </motion.div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <div className="chat-input-container">
                        <input
                            className="chat-input"
                            placeholder="Ask about inverter health, SHAP features, maintenance..."
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            disabled={loading}
                        />
                        <button className="chat-send-btn" onClick={() => sendMessage()} disabled={loading}>
                            <Send size={18} />
                        </button>
                    </div>

                    {/* Quick Actions */}
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
