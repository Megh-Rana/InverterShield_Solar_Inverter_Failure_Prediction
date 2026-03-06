"""
GenAI LLM Module for Solar Inverter Failure Prediction.

Provides natural language summaries and root-cause hypotheses using:
- Primary: Google Gemini 1.5 Flash API
- Fallback: Local Ollama (llama3/mistral)
"""

import os
import json
import asyncio
from typing import Tuple, Optional, Dict


SYSTEM_PROMPT = """You are an expert solar energy engineer and AI assistant for a solar inverter monitoring platform.

Your role is to:
1. Analyze inverter telemetry data and provide clear, actionable insights
2. Explain ML model predictions in plain language
3. Suggest root causes for failures and degradation
4. Recommend maintenance actions based on risk levels and SHAP feature importance

Key domain knowledge:
- Solar inverters convert DC (from panels) to AC (for grid)
- Common failure modes: overheating, grid frequency deviation, PV string failures, alarm code triggers
- Risk levels: 0=No Risk, 1=Degradation Risk (performance drop), 2=Shutdown Risk (imminent failure)
- SHAP values indicate which features most influence predictions

Always cite specific data values when available. Be concise and actionable.
Do NOT hallucinate data — only reference values provided in the context.
"""


async def _try_gemini(message: str, context: Optional[Dict] = None) -> Optional[str]:
    """Try Google Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        # Build context-enriched prompt
        full_prompt = SYSTEM_PROMPT + "\n\n"
        if context:
            full_prompt += f"Context data:\n```json\n{json.dumps(context, indent=2, default=str)}\n```\n\n"
        full_prompt += f"User question: {message}"

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


async def _try_ollama(message: str, context: Optional[Dict] = None) -> Optional[str]:
    """Try local Ollama with llama3 or mistral."""
    try:
        import httpx

        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

        full_prompt = ""
        if context:
            full_prompt += f"Context data:\n```json\n{json.dumps(context, indent=2, default=str)}\n```\n\n"
        full_prompt += f"User question: {message}"

        # Try llama3 first, then mistral
        for model in ["llama3", "mistral"]:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        f"{ollama_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": full_prompt,
                            "system": SYSTEM_PROMPT,
                            "stream": False,
                        }
                    )
                    if response.status_code == 200:
                        return response.json().get("response", "")
            except Exception:
                continue

        return None
    except Exception as e:
        print(f"Ollama error: {e}")
        return None


async def get_llm_response(
    message: str,
    inverter_id: Optional[str] = None,
    context: Optional[Dict] = None,
) -> Tuple[str, str]:
    """
    Get LLM response with fallback chain: Gemini → Ollama → Rule-based.

    Returns:
        (response_text, source) where source is "gemini", "ollama", or "fallback"
    """
    # Enrich context with inverter ID if provided
    if context is None:
        context = {}
    if inverter_id:
        context["inverter_id"] = inverter_id

    # Try Gemini first
    result = await _try_gemini(message, context)
    if result:
        return result, "gemini"

    # Fallback to Ollama
    result = await _try_ollama(message, context)
    if result:
        return result, "ollama"

    # Final fallback: rule-based
    return _rule_based_response(message, context), "fallback"


def _rule_based_response(message: str, context: Optional[Dict] = None) -> str:
    """Rule-based fallback when no LLM is available."""
    msg_lower = message.lower()

    if any(w in msg_lower for w in ["why", "cause", "root cause", "reason"]):
        return (
            "Based on the model's analysis, the most common root causes for inverter issues include:\n"
            "1. **Temperature stress** — prolonged high inverter temperatures reduce component lifespan\n"
            "2. **PV string degradation** — zero or low-current strings indicate panel/wiring issues\n"
            "3. **Grid instability** — frequency deviations from 50Hz stress the inverter's synchronization\n"
            "4. **Accumulated alarms** — repeated alarm events signal developing faults\n\n"
            "For a specific diagnosis, please provide the inverter telemetry data."
        )

    if any(w in msg_lower for w in ["recommend", "action", "what should", "maintenance"]):
        return (
            "**Recommended maintenance actions:**\n"
            "- **No Risk (Level 0):** Continue routine monitoring. Schedule next inspection per normal cycle.\n"
            "- **Degradation Risk (Level 1):** Inspect within 7 days. Check PV string connections, clean panels, verify power factor.\n"
            "- **Shutdown Risk (Level 2):** Immediate inspection required. Check alarm codes, inverter temperature, grid connection stability.\n\n"
            "Always correlate findings with local weather data and grid conditions."
        )

    return (
        "I'm the Solar Inverter AI Assistant. I can help with:\n"
        "- Explaining failure predictions and risk levels\n"
        "- Identifying root causes from SHAP analysis\n"
        "- Recommending maintenance actions\n"
        "- Analyzing inverter performance trends\n\n"
        "Try asking: 'What are the top risk factors?' or 'Why is inverter X showing degradation?'"
    )


def generate_inverter_summary(prediction: Dict, shap_factors: list) -> str:
    """Generate a natural language summary for an inverter prediction."""
    inv_id = prediction.get("inverter_id", "Unknown")
    risk = prediction.get("risk_level", "unknown")
    prob = prediction.get("failure_probability", 0)

    summary = f"**Inverter {inv_id}** — "

    if risk == "no_risk":
        summary += f"Operating normally (failure probability: {prob:.1%}). "
    elif risk == "degradation_risk":
        summary += f"⚠️ Degradation risk detected (failure probability: {prob:.1%}). "
    else:
        summary += f"🚨 HIGH SHUTDOWN RISK (failure probability: {prob:.1%}). "

    if shap_factors:
        top = shap_factors[:3]
        drivers = ", ".join([f"{f['feature']} ({f['direction']})" for f in top])
        summary += f"Key drivers: {drivers}."

    if prediction.get("is_anomaly"):
        summary += " ⚠️ Anomalous behavior detected by Isolation Forest."

    return summary
