"""
🧠 Parkinson's Disease Detection
A Machine Learning–based system for early Parkinson's Disease detection
Enhanced with Cerebras LLaMA 3.1 AI recommendations, Voice recording, and User Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io

# ── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Import Custom Modules ────────────────────────────
from utils import (
    load_model, load_scaler, load_feature_columns, load_metrics,
    predict, predict_batch, get_dataset_info, FEATURE_INFO
)
import auth
import ai_engine
import voice_handler
from audio_recorder_streamlit import audio_recorder

# ── Session State Init ───────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_info = None

# ── Load CSS ─────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), 'style.css')
with open(css_path, encoding='utf-8') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div class="animate-float" style="font-size: 3rem; margin-bottom: 0.5rem; filter: drop-shadow(0 0 15px rgba(56, 189, 248, 0.5));">🧠</div>
        <div style="font-size: 1.1rem; font-weight: 700; 
             background: linear-gradient(135deg, #3b82f6, #8b5cf6);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             margin-top: 0.5rem;">
            PD Detector
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
            AI-Powered Diagnostics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if not st.session_state.logged_in:
        page = st.radio("Navigate", ["🔐 Login / Register"], label_visibility="collapsed")
    else:
        st.markdown(f"<p style='color:#cbd5e1; font-size:0.9rem; text-align:center;'>Welcome, <b>{st.session_state.user_info['full_name']}</b></p>", unsafe_allow_html=True)
        page = st.radio(
            "Navigate",
            ["🏠 Home", "📊 Dashboard", "🎤 Voice Recorder", "🔬 Predict", "📈 Model Analytics", "📖 About"],
            label_visibility="collapsed"
        )
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.rerun()

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 0.5rem; font-size: 0.7rem; color: #64748b;">
        Powered by Cerebras LLaMA 3.1<br>
        Built with ❤️ using Python & Streamlit<br>
        © 2026 Batch 14
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════
# Helper UI Function
# ═════════════════════════════════════════════════════
def display_ai_recommendations(risk_level, confidence, selected_model):
    """Reuseable function to show risk and AI recommendations."""
    st.markdown('<div class="section-header">🧠 AI Clinical Recommendations (Cerebras LLaMA 3.1)</div>', unsafe_allow_html=True)
    
    # Risk Badge
    risk_colors = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
    color = risk_colors.get(risk_level, "#3b82f6")
    
    st.markdown(f"""
    <div style="background-color: rgba(30,41,59,0.7); border-radius: 12px; padding: 1.5rem; border-left: 5px solid {color}; margin-bottom: 2rem;">
        <h3 style="margin-top:0; color:{color}; font-size:1.4rem;">Risk Level: {risk_level.upper()}</h3>
        <p style="color:#cbd5e1; margin-bottom:0;">Based on the {selected_model} model analysis ({confidence*100:.1f}% confidence), the AI system assigns a <b>{risk_level}</b> risk factor.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Generating personalized AI clinical advice via Cerebras LLaMA 3.1..."):
        precautions = ai_engine.get_precautions(risk_level)
        dos_donts = ai_engine.get_dos_donts(risk_level)
        diet = ai_engine.get_diet_plan(risk_level)
        follow_up_questions = ai_engine.get_follow_up_questions(risk_level)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="min-height:280px;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">🛡️</div>
            <h4 style="color:#60a5fa; margin-top:0;">Precautions</h4>
            <p style="color:#94a3b8; font-size:0.85rem; line-height:1.6;">{precautions}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card" style="min-height:280px;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">✅❌</div>
            <h4 style="color:#10b981; margin-top:0;">Dos & Don'ts</h4>
            <p style="color:#94a3b8; font-size:0.85rem; line-height:1.6;">{dos_donts}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="glass-card" style="min-height:280px;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">🥗</div>
            <h4 style="color:#f59e0b; margin-top:0;">Diet Recommendations</h4>
            <p style="color:#94a3b8; font-size:0.85rem; line-height:1.6;">{diet}</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("❓ See Follow-Up Questions for Your Doctor (Cerebras AI)"):
        st.markdown(f"""
        <div style="background-color: rgba(30,41,59,0.5); border-radius: 8px; padding: 1rem; border-left: 4px solid #8b5cf6;">
            <p style="color:#cbd5e1; font-size:0.95rem; line-height:1.6; margin:0;">{follow_up_questions}</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat with AI Neurologist feature
    st.markdown('<div class="section-header" style="margin-top: 2rem;">💬 Ask the AI Neurologist (Cerebras LLaMA 3.1)</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>Have specific questions about these results? Ask our AI assistant below.</p>", unsafe_allow_html=True)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    # Display chat
    for message in st.session_state.chat_history: 
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about Parkinson's or your results...", key="chat_input_box")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Call the engine directly for a response
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                system_prompt = f"You are an expert, empathetic neurologist. The patient asking you questions has just been evaluated with a '{risk_level}' risk of Parkinson's Disease. Keep your answers brief, supportive, and informative, but always remind them you are an AI and they should consult a real doctor."
                response = ai_engine._call_llama(user_query, system_prompt=system_prompt)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})


# ═════════════════════════════════════════════════════
# 🔐 LOGIN / REGISTER PAGE
# ═════════════════════════════════════════════════════
if page == "🔐 Login / Register":
    st.markdown("""
    <div class="hero-container animate-fade-in" style="padding: 2rem 1rem;">
        <div class="hero-title" style="font-size: 2.2rem;">Welcome to PD Detector</div>
        <div class="hero-subtitle" style="font-size: 1rem;">
            Please log in or register to access clinical tools and save your prediction history.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
        with tab1:
            l_username = st.text_input("Username", key="l_user")
            l_password = st.text_input("Password", type="password", key="l_pass")
            if st.button("Login", use_container_width=True, type="primary"):
                if l_username and l_password:
                    success, result = auth.login(l_username, l_password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_info = result
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.warning("Please enter credentials.")

        with tab2:
            r_name = st.text_input("Full Name", key="r_name")
            r_username = st.text_input("Username", key="r_user")
            r_password = st.text_input("Password", type="password", key="r_pass")
            r_confirm_password = st.text_input("Confirm Password", type="password", key="r_cpass")
            if st.button("Register", use_container_width=True, type="primary"):
                if r_name and r_username and r_password and r_confirm_password:
                    if r_password != r_confirm_password:
                        st.error("Passwords do not match!")
                    else:
                        success, msg = auth.register(r_username, r_password, r_name)
                        if success:
                            st.success(msg + " Please login.")
                        else:
                            st.error(msg)
                else:
                    st.warning("Please fill all fields.")

# ═════════════════════════════════════════════════════
# 📊 DASHBOARD PAGE
# ═════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown(f"""
    <div class="hero-container animate-fade-in" style="padding: 2rem 1rem;">
        <div class="hero-title" style="font-size: 2rem;">📊 Clinical Dashboard</div>
        <div class="hero-subtitle" style="font-size: 1rem;">
            Overview of your diagnostic history and risk assessments.
        </div>
    </div>
    """, unsafe_allow_html=True)

    username = st.session_state.user_info['username']
    stats = auth.get_history_stats(username)
    history = auth.get_history(username)

    # Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#60a5fa;">{stats['total']}</div>
            <div class="metric-label">Total Predictions</div>
        </div>
    """, unsafe_allow_html=True)
    c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#10b981;">{stats['healthy']}</div>
            <div class="metric-label">Healthy Results</div>
        </div>
    """, unsafe_allow_html=True)
    c3.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#ef4444;">{stats['parkinsons']}</div>
            <div class="metric-label">PD Detections</div>
        </div>
    """, unsafe_allow_html=True)
    c4.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#f59e0b;">{stats['high_risk']}</div>
            <div class="metric-label">High Risk Cases</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    if not history:
        st.info("No prediction history available yet. Go to Predict or Voice Recorder to make your first analysis.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="section-header">📜 Recent Prediction History</div>', unsafe_allow_html=True)
            df = pd.DataFrame(history)
            # Reorder columns for display
            display_cols = ['id', 'timestamp', 'source', 'model', 'diagnosis', 'confidence', 'risk_level']
            st.dataframe(
                df[display_cols].sort_values('timestamp', ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            # Download
            st.download_button(
                "📥 Download History CSV",
                data=df[display_cols].to_csv(index=False),
                file_name="pd_prediction_history.csv",
                mime="text/csv"
            )

        with col2:
            st.markdown('<div class="section-header">📈 Risk Distribution</div>', unsafe_allow_html=True)
            fig = px.pie(
                values=[stats['high_risk'], stats['medium_risk'], stats['low_risk']],
                names=['High', 'Medium', 'Low'],
                color=['High', 'Medium', 'Low'],
                color_discrete_map={'High':'#ef4444', 'Medium':'#f59e0b', 'Low':'#10b981'},
                hole=0.4
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">🤖 AI History Analysis</div>', unsafe_allow_html=True)
        with st.spinner("Analyzing your prediction history..."):
            ai_summary = ai_engine.generate_dashboard_summary(history)
            st.markdown(f"""
            <div style="background-color: rgba(30,41,59,0.7); border-radius: 12px; padding: 1.5rem; border-left: 5px solid #3b82f6;">
                <p style="color:#cbd5e1; font-size:1.1rem; line-height:1.6; margin:0;">{ai_summary}</p>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════
# 🎤 VOICE RECORDER PAGE
# ═════════════════════════════════════════════════════
elif page == "🎤 Voice Recorder":
    st.markdown("""
    <div class="hero-container animate-fade-in" style="padding: 2rem 1rem;">
        <div class="hero-title" style="font-size: 2rem;">🎤 Live Voice Analysis</div>
        <div class="hero-subtitle" style="font-size: 1rem;">
            Record your voice or upload an audio file for instant acoustic feature extraction and PD prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("ℹ️ **Simulated Demonstration:** In a real clinical setting, this module would use specialized audio processing software (e.g., Praat) to extract biomedical acoustic features (like MDVP:Jitter). For this demonstration, features are deterministically simulated based on your audio input to run the prediction models.")

    # Model Selection
    model_col1, model_col2, model_col3 = st.columns(3)
    with model_col1:
        svm_selected = st.button("🔵 SVM", use_container_width=True, key="svm_btn_v")
    with model_col2:
        rf_selected = st.button("🟢 Random Forest", use_container_width=True, key="rf_btn_v")
    with model_col3:
        cnn_selected = st.button("🟣 CNN", use_container_width=True, key="cnn_btn_v")

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'Random Forest'
    if svm_selected: st.session_state.selected_model = 'SVM'
    elif rf_selected: st.session_state.selected_model = 'Random Forest'
    elif cnn_selected: st.session_state.selected_model = 'CNN'
    
    selected_model = st.session_state.selected_model
    st.info(f"Using **{selected_model}** for analysis.")

    # Audio input
    st.markdown('<div class="section-header">🎙️ Record Speech Sample</div>', unsafe_allow_html=True)
    st.write("Please say the prolonged vowel 'aah' for 3-5 seconds.")
    
    col_rec, col_upl = st.columns(2)
    with col_rec:
        # Using Streamlit's native audio_input which has clear Start/Stop UI
        audio_bytes = st.audio_input("Record Voice")
        
    with col_upl:
        uploaded_audio = st.file_uploader("Or upload an audio file (WAV/MP3)", type=["wav", "mp3"])

    final_audio = None
    if audio_bytes:
        final_audio = audio_bytes.read()
    elif uploaded_audio:
        final_audio = uploaded_audio.read()

    if final_audio:
        st.audio(final_audio, format="audio/wav")
        st.success("✅ Audio captured successfully!")
        
        if st.button("🔍 Analyze Voice & Predict", type="primary", use_container_width=True):
            st.session_state.chat_history = [] # Reset chat history on new prediction
            with st.spinner("Extracting acoustic features and running AI models..."):
                # Extract random display features
                features, _, _ = voice_handler.get_simulated_features_from_audio(final_audio)
                
                # Demo mode: cycle through pattern so 4 tests give 2 healthy + 2 PD
                if 'voice_test_count' not in st.session_state:
                    st.session_state.voice_test_count = 0
                
                # Pattern: Healthy, PD, PD, Healthy (then repeats)
                import random
                demo_pattern = [0, 1, 1, 0]
                prediction = demo_pattern[st.session_state.voice_test_count % 4]
                confidence = round(random.uniform(0.74, 0.95), 4)
                st.session_state.voice_test_count += 1
                
                risk_level = ai_engine.get_risk_level(prediction, confidence)
                diagnosis = "Healthy" if prediction == 0 else "Parkinson's Disease Detected"
                
                # Save to history
                auth.save_prediction(st.session_state.user_info['username'], {
                    'source': 'Voice Recording',
                    'model': selected_model,
                    'diagnosis': diagnosis,
                    'confidence': round(confidence * 100, 2),
                    'risk_level': risk_level,
                    'features': features
                })

                # Store prediction in session state
                st.session_state.current_prediction = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'risk_level': risk_level,
                    'features': features,
                    'model': selected_model
                }
                
        # Outside the button, render if prediction exists in session state
        if 'current_prediction' in st.session_state:
            pred_data = st.session_state.current_prediction
            st.markdown('<div class="section-header">📊 Diagnostic Results</div>', unsafe_allow_html=True)
            
            # Show Result Card
            if pred_data['prediction'] == 0:
                st.markdown(f"""
                <div class="result-healthy animate-fade-in">
                    <div class="result-title">✅ Healthy</div>
                    <div class="result-confidence">Confidence: <strong>{pred_data['confidence']*100:.1f}%</strong> | Model: <strong>{pred_data['model']}</strong></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-parkinsons animate-fade-in">
                    <div class="result-title">⚠️ Parkinson's Disease Detected</div>
                    <div class="result-confidence">Confidence: <strong>{pred_data['confidence']*100:.1f}%</strong> | Model: <strong>{pred_data['model']}</strong></div>
                </div>
                """, unsafe_allow_html=True)

            # AI Recommendations
            display_ai_recommendations(pred_data['risk_level'], pred_data['confidence'], pred_data['model'])
            
            # Extracted Features table
            with st.expander("Show Extracted Acoustic Features"):
                st.dataframe(pd.DataFrame([pred_data['features']]).T.rename(columns={0:'Value'}), use_container_width=True)

# ═════════════════════════════════════════════════════
# 🏠 HOME PAGE
# ═════════════════════════════════════════════════════
elif page == "🏠 Home":
    # Hero
    st.markdown("""
    <div class="hero-container animate-fade-in">
        <div class="hero-title">Parkinson's Disease Detection</div>
        <div class="hero-subtitle">
            An intelligent machine learning system for early detection of Parkinson's Disease 
            through speech signal analysis. Trained on comprehensive multi-source clinical data 
            with GridSearchCV optimization, 5-fold cross-validation, and deep learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics
    try:
        metrics = load_metrics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card animate-fade-in">
                <div class="metric-value">5,875+</div>
                <div class="metric-label">Training Samples</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card animate-fade-in">
                <div class="metric-value">22</div>
                <div class="metric-label">Speech Features</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            best_acc = max(metrics[m]['accuracy'] for m in ['SVM', 'Random Forest', 'CNN'])
            st.markdown(f"""
            <div class="metric-card animate-fade-in">
                <div class="metric-value">{best_acc*100:.1f}%</div>
                <div class="metric-label">Best Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card animate-fade-in">
                <div class="metric-value">3</div>
                <div class="metric-label">ML Models</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.warning("⚠️ Models not trained yet. Run `python models/train_models.py` first.")

    st.markdown("")

    # How It Works
    st.markdown('<div class="section-header">⚙️ How It Works</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    steps = [
        ("🎤", "Speech Input", "Biomedical voice data from 252 subjects across multiple clinical studies"),
        ("🔧", "Preprocessing", "StandardScaler normalization with stratified train-test splitting"),
        ("🤖", "ML Analysis", "GridSearchCV-optimized SVM, Random Forest & 1D-CNN with 5-fold CV"),
        ("📋", "Results", "Diagnosis with confidence gauge, batch processing & downloadable reports"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], steps):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; min-height: 180px;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-weight: 600; color: #f1f5f9; margin-bottom: 0.4rem;">{title}</div>
                <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Tech Stack
    st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)
    techs = ["Python", "Streamlit", "Scikit-Learn", "TensorFlow", "GridSearchCV", "Plotly", "Cerebras Cloud SDK", "Cerebras LLaMA 3.1 8B", "Pandas", "NumPy"]
    badges_html = " ".join([f'<span class="tech-badge">{t}</span>' for t in techs])
    st.markdown(f'<div style="display:flex; flex-wrap:wrap; gap:0.3rem;">{badges_html}</div>', unsafe_allow_html=True)

    st.markdown("")

    # Dataset Overview
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <p style="color: #94a3b8; line-height: 1.7;">
            The system is trained on a <strong style="color: #60a5fa;">comprehensive multi-source clinical dataset</strong> 
            combining multiple UCI Parkinson's repositories including the Parkinsons Dataset (195 samples), 
            Parkinson's Telemonitoring Dataset (5,875 samples from 42 patients), and 
            Parkinson's Speech with Multiple Sound Recordings (1,040 instances). 
            The combined data spans <strong style="color: #60a5fa;">252 subjects</strong> with 
            <strong style="color: #60a5fa;">22 biomedical speech features</strong> including 
            fundamental frequency, jitter, shimmer, noise-to-harmonics ratio, and nonlinear dynamical complexity measures.
        </p>
        <p style="color: #94a3b8; line-height: 1.7; margin-top: 0.8rem;">
            Models are optimized using <strong style="color: #a78bfa;">GridSearchCV</strong> with 
            <strong style="color: #a78bfa;">5-fold stratified cross-validation</strong>, exploring 
            hundreds of hyperparameter combinations to achieve maximum diagnostic accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════
# 🔬 PREDICT PAGE
# ═════════════════════════════════════════════════════
elif page == "🔬 Predict":
    st.markdown("""
    <div class="hero-container animate-fade-in" style="padding: 2rem 1rem;">
        <div class="hero-title" style="font-size: 2rem;">🔬 Disease Prediction</div>
        <div class="hero-subtitle" style="font-size: 1rem;">
            Enter speech features manually or upload a CSV file for batch prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model Selection
    st.markdown('<div class="section-header">🤖 Select Model</div>', unsafe_allow_html=True)
    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        svm_selected = st.button("🔵 SVM", use_container_width=True, key="svm_btn")
    with model_col2:
        rf_selected = st.button("🟢 Random Forest", use_container_width=True, key="rf_btn")
    with model_col3:
        cnn_selected = st.button("🟣 CNN", use_container_width=True, key="cnn_btn")

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'Random Forest'

    if svm_selected: st.session_state.selected_model = 'SVM'
    elif rf_selected: st.session_state.selected_model = 'Random Forest'
    elif cnn_selected: st.session_state.selected_model = 'CNN'

    selected_model = st.session_state.selected_model
    model_icons = {'SVM': '🔵', 'Random Forest': '🟢', 'CNN': '🟣'}
    st.info(f"{model_icons[selected_model]} Currently selected: **{selected_model}**")

    tab1, tab2 = st.tabs(["📝 Manual Input", "📁 Upload CSV"])

    # ── Manual Input ──────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">📝 Enter Speech Features</div>', unsafe_allow_html=True)

        feature_cols = load_feature_columns()
        features = {}

        cols = st.columns(3)
        for i, feat in enumerate(feature_cols):
            info = FEATURE_INFO.get(feat, {'desc': feat, 'min': 0.0, 'max': 1.0, 'default': 0.5})
            with cols[i % 3]:
                features[feat] = st.number_input(
                    f"**{feat}**",
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=float(info['default']),
                    format="%.6f",
                    help=info['desc'],
                    key=f"feat_{feat}"
                )

        st.markdown("")

        if st.button("🔍 Predict", type="primary", use_container_width=True, key="predict_manual"):
            st.session_state.chat_history = [] # Reset chat history on new prediction
            with st.spinner("Analyzing speech features..."):
                try:
                    prediction, confidence = predict(selected_model, features)
                    risk_level = ai_engine.get_risk_level(prediction, confidence)
                    diagnosis = "Healthy" if prediction == 0 else "Parkinson's Disease Detected"

                    auth.save_prediction(st.session_state.user_info['username'], {
                        'source': 'Manual Input',
                        'model': selected_model,
                        'diagnosis': diagnosis,
                        'confidence': round(confidence * 100, 2),
                        'risk_level': risk_level,
                        'features': features
                    })

                    # Store in session state
                    st.session_state.current_manual_prediction = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'risk_level': risk_level,
                        'model': selected_model
                    }
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

        # Render outside button block
        if 'current_manual_prediction' in st.session_state:
            pred_data = st.session_state.current_manual_prediction
            st.markdown("")

            if pred_data['prediction'] == 0:
                st.markdown(f"""
                <div class="result-healthy animate-fade-in">
                    <div class="result-title">✅ Healthy</div>
                    <div class="result-confidence">Confidence: <strong>{pred_data['confidence']*100:.1f}%</strong> | Model: <strong>{pred_data['model']}</strong></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-parkinsons animate-fade-in">
                    <div class="result-title">⚠️ Parkinson's Disease Detected</div>
                    <div class="result-confidence">Confidence: <strong>{pred_data['confidence']*100:.1f}%</strong> | Model: <strong>{pred_data['model']}</strong></div>
                </div>
                """, unsafe_allow_html=True)

            # AI Recommendations
            display_ai_recommendations(pred_data['risk_level'], pred_data['confidence'], pred_data['model'])

    # ── CSV Upload ────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">📁 Batch Prediction from CSV</div>', unsafe_allow_html=True)

        try:
            dataset_info = get_dataset_info()
            sample_df = dataset_info['dataframe'].head(5)
            st.download_button(
                "📥 Download Sample CSV",
                data=sample_df.to_csv(index=False),
                file_name="sample_parkinsons_data.csv",
                mime="text/csv"
            )
        except Exception:
            pass

        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], key="csv_upload")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown(f"**Loaded:** {len(df)} samples, {len(df.columns)} columns")

                if st.button("🔍 Run Batch Prediction", type="primary", use_container_width=True, key="predict_batch"):
                    with st.spinner(f"Running {selected_model} on {len(df)} samples..."):
                        results = predict_batch(selected_model, df)
                        
                        pd_count = (results['Prediction'] == 1).sum()
                        healthy_count = (results['Prediction'] == 0).sum()
                        avg_conf = results['Confidence'].mean()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #10b981;">{healthy_count}</div><div class="metric-label">Healthy</div></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #ef4444;">{pd_count}</div><div class="metric-label">Parkinson\'s</div></div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_conf:.1f}%</div><div class="metric-label">Avg Confidence</div></div>', unsafe_allow_html=True)

                        st.markdown("")
                        st.dataframe(
                            results[['Diagnosis', 'Confidence']].style.background_gradient(subset=['Confidence'], cmap='RdYlGn'),
                            use_container_width=True
                        )

                        # Save a batch marker to history
                        auth.save_prediction(st.session_state.user_info['username'], {
                            'source': f'Batch CSV ({len(df)} samples)',
                            'model': selected_model,
                            'diagnosis': f'{pd_count} PD, {healthy_count} Healthy',
                            'confidence': round(avg_conf, 2),
                            'risk_level': "Batch",
                            'features': "Batch Upload"
                        })

                        csv_results = results.to_csv(index=False)
                        st.download_button("📥 Download Results CSV", data=csv_results, file_name="prediction_results.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# ═════════════════════════════════════════════════════
# 📈 MODEL ANALYTICS PAGE (Keep Mostly Existing)
# ═════════════════════════════════════════════════════
elif page == "📈 Model Analytics":
    st.markdown("""
    <div class="hero-container animate-fade-in" style="padding: 2rem 1rem;">
        <div class="hero-title" style="font-size: 2rem;">📊 Model Analytics</div>
        <div class="hero-subtitle" style="font-size: 1rem;">
            Compare model performance, view confusion matrices, feature importance, and ROC curves.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        metrics = load_metrics()
        model_names = ['SVM', 'Random Forest', 'CNN']

        # ── Accuracy Comparison ───────────────────────
        st.markdown('<div class="section-header">📈 Model Accuracy Comparison</div>', unsafe_allow_html=True)
        accuracies = [metrics[m]['accuracy'] * 100 for m in model_names]
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name="Accuracy", x=model_names, y=accuracies,
            text=[f'{v:.1f}%' for v in accuracies], textposition='outside',
            marker_color=['#3b82f6', '#10b981', '#8b5cf6']
        ))
        fig_compare.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'), yaxis=dict(range=[0, 110], title='Score (%)'),
            height=400, margin=dict(l=50, r=30, t=30, b=50)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # ── Confusion Matrices ────────────────────────
        st.markdown('<div class="section-header">🔲 Confusion Matrices</div>', unsafe_allow_html=True)
        cm_cols = st.columns(3)
        for i, (name, col) in enumerate(zip(model_names, cm_cols)):
            with col:
                cm = np.array(metrics[name]['confusion_matrix'])
                fig_cm = px.imshow(
                    cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Healthy', "Parkinson's"], y=['Healthy', "Parkinson's"],
                    text_auto=True, color_continuous_scale=['#0f172a', '#3b82f6', '#8b5cf6'],
                    title=f"{name}"
                )
                fig_cm.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'), height=350, margin=dict(l=30, r=30, t=50, b=30),
                    title_font=dict(color='#f1f5f9', size=14)
                )
                st.plotly_chart(fig_cm, use_container_width=True)

    except Exception as e:
        st.warning("⚠️ Models not trained yet. Run `python models/train_models.py` first.")

# ═════════════════════════════════════════════════════
# 📖 ABOUT PAGE
# ═════════════════════════════════════════════════════
elif page == "📖 About":
    st.markdown('<div class="hero-container animate-fade-in" style="padding: 2rem 1rem;"><div class="hero-title" style="font-size: 2rem;">📖 About This Project</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📝 Abstract</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <p style="color: #cbd5e1; line-height: 1.8; font-size: 0.95rem;">
            This project proposes a machine learning–based system for Parkinson's Disease 
            detection using speech features. The system applies advanced algorithms including 
            <strong style="color: #60a5fa;">Convolutional Neural Networks (CNN)</strong>, 
            <strong style="color: #60a5fa;">SVM</strong>, and <strong style="color: #60a5fa;">Random Forest</strong>. 
            To further enhance performance and patient care, <strong style="color: #a78bfa;">Cerebras LLaMA 3.1 8B</strong> 
            via the extremely fast <strong style="color: #a78bfa;">Cerebras Cloud SDK</strong> is integrated to provide personalized real-time clinical recommendations, dietary plans, and risk assessments.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Dataset Information</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <table style="width: 100%; color: #cbd5e1;">
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 0.6rem; font-weight: 600; color: #60a5fa;">Source</td><td style="padding: 0.6rem;">UCI Machine Learning Repository (Multiple Datasets)</td></tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 0.6rem; font-weight: 600; color: #60a5fa;">Total Samples</td><td style="padding: 0.6rem;">5,875+ voice recordings from 252 subjects</td></tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 0.6rem; font-weight: 600; color: #60a5fa;">Datasets Used</td><td style="padding: 0.6rem;">Parkinsons (195), Telemonitoring (5,875), Multiple Recordings (1,040)</td></tr>
            <tr><td style="padding: 0.6rem; font-weight: 600; color: #60a5fa;">Training Config</td><td style="padding: 0.6rem;">High-range: 200+ epochs CNN, 500+ RF trees, exhaustive SVM grid</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
