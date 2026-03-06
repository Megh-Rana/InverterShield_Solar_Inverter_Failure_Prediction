import { useState } from 'react'
import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, Activity, BarChart3, MessageSquare,
  Menu, X, Zap, Sun
} from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Prediction from './pages/Prediction'
import Performance from './pages/Performance'
import Chat from './pages/Chat'

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/predict', label: 'Live Prediction', icon: Activity },
  { path: '/performance', label: 'Model Metrics', icon: BarChart3 },
  { path: '/chat', label: 'AI Copilot', icon: MessageSquare },
]

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()

  return (
    <div className="app-layout">
      {/* Mobile Header */}
      <header className="mobile-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="brand-icon"><Sun size={14} color="white" /></div>
          <span style={{ fontSize: 14, fontWeight: 600 }}>SolarGuard</span>
        </div>
        <button className="hamburger" onClick={() => setSidebarOpen(!sidebarOpen)}>
          {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </header>

      {/* Sidebar Overlay */}
      <div
        className={`sidebar-overlay ${sidebarOpen ? 'visible' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-brand">
          <div className="brand-icon"><Sun size={16} color="white" /></div>
          <div>
            <h1>InverterShield</h1>
            <span>Predictive Maintenance</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {navItems.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              end={path === '/'}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              onClick={() => setSidebarOpen(false)}
            >
              <Icon />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <p>XGBoost · SHAP · Gemini AI<br />Binary F1: 0.918 · AUC: 0.901</p>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <div className="page-container">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Prediction />} />
            <Route path="/performance" element={<Performance />} />
            <Route path="/chat" element={<Chat />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}
