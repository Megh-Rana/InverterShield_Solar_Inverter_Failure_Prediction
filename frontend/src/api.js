const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function fetchHealth() {
    const res = await fetch(`${API_URL}/health`);
    return res.json();
}

export async function fetchDashboard() {
    const res = await fetch(`${API_URL}/dashboard`);
    return res.json();
}

export async function fetchModelInfo() {
    const res = await fetch(`${API_URL}/model/info`);
    return res.json();
}

export async function postPredict(features, inverterId = 'manual') {
    const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features, inverter_id: inverterId }),
    });
    return res.json();
}

export async function postChat(message, context = null) {
    const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, context }),
    });
    return res.json();
}
