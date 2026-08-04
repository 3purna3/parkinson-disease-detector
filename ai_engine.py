"""
Cerebras AI Inference Engine for Parkinson's Disease Recommendations
Uses llama3.1-8b via Cerebras Cloud SDK to generate clinical advice.
"""

import os
from cerebras.cloud.sdk import Cerebras

# Try to get from environment first, fallback to hardcoded (for demo purposes)
API_KEY = os.environ.get("CEREBRAS_API_KEY", "csk-wymhfvv9mrnx43nft5ky5e96pw6ceynhe25hmd6rerctf6xv")

# Initialize client
try:
    client = Cerebras(api_key=API_KEY)
except Exception as e:
    print(f"Failed to initialize Cerebras client: {e}")
    client = None

def get_risk_level(prediction, confidence):
    """Determine risk level based on model prediction and confidence."""
    if prediction == 0:
        return "Low"
    elif confidence >= 0.85:
        return "High"
    else:
        return "Medium"

def _call_llama(prompt, system_prompt="You are an expert neurologist specializing in Parkinson's Disease."):
    """Helper to call Cerebras LLaMA 3.1 8B."""
    if not client:
        return "Error: AI engine not initialized. Please set the CEREBRAS_API_KEY environment variable."
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="llama3.1-8b",
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Generation Failed: {str(e)}\n\n(Fallback: Please consult a healthcare professional for personalized advice.)"

def get_precautions(risk_level):
    """Generate precautions based on risk level."""
    prompt = f"A patient's voice analysis indicates a {risk_level} risk for Parkinson's Disease. Briefly list 4-5 immediate clinical precautions they should take. Keep it concise, practical, and medical-focused."
    return _call_llama(prompt)

def get_dos_donts(risk_level):
    """Generate specific Dos and Don'ts."""
    prompt = f"For a patient with a {risk_level} risk of Parkinson's Disease, provide a concise list of 3 'Dos' and 3 'Don'ts' for their daily life and safety."
    return _call_llama(prompt)

def get_diet_plan(risk_level):
    """Generate a dietary recommendation plan."""
    prompt = f"Recommend a brief, evidence-based dietary plan for someone with a {risk_level} risk of Parkinson's Disease. Mention specific foods to include and avoid. Keep it under 100 words."
    return _call_llama(prompt)

def get_follow_up_questions(risk_level):
    """Generate follow-up questions to ask a doctor."""
    prompt = f"A patient's voice analysis indicates a {risk_level} risk for Parkinson's Disease. Briefly list 3 essential follow-up questions they should ask their neurologist or primary care physician during their next visit."
    return _call_llama(prompt)

def generate_dashboard_summary(history_data: list):
    """Generate a brief, user-friendly summary of the patient's entire prediction history."""
    if not history_data:
        return "No prediction history available to summarize."
        
    recent_entries = history_data[:5] # Analyze last 5 predictions
    history_str = "\n".join([f"- Date: {e.get('timestamp', 'Unknown')}, Risk: {e.get('risk_level', 'Unknown')}, Confidence: {e.get('confidence', 'Unknown')}%" for e in recent_entries])
    
    prompt = f"Here are the last {len(recent_entries)} Parkinson's Disease voice analysis predictions for a patient:\n{history_str}\n\nProvide a very brief, empathetic, 2-sentence summary of this trend for the patient's dashboard. Avoid medical diagnoses, focus on the trend (e.g., 'Your recent tests show consistently low risk...')."
    return _call_llama(prompt)
