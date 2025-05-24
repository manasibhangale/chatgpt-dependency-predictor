import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load the trained pipeline
with open("chatgpt_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature names matching training
feature_names = [
    "chatgpt_usage_frequency_per_week",
    "average_duration_per_session_minutes",
    "attempt_before_chatgpt",
    "confidence_in_solving_alone",
    "peer_usage_influence",
    "cgpa",
    "used_other_ai_tools",
    "chatgpt_preferred_over_google",
    "reason_for_using_chatgpt",
    "department"
]

# Custom CSS for classy dark theme with blue accents
st.markdown(
    """
    <style>
    .reportview-container, .main {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .css-10trblm h1 {
        font-weight: 700;
        font-size: 3rem;
        color: #8a97a5;
        margin-bottom: 0.3rem;
    }
    #MainMenu, footer, header {
        visibility: hidden;
    }
    .css-1d391kg {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        background: none;
    }
    label, .stNumberInput label, .stSelectbox label, .stRadio label {
        color: #b0b8c1;
        font-weight: 600;
    }
    input[type=number], div[role="listbox"] {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
        border: 1px solid #3a3f51 !important;
        border-radius: 4px !important;
        padding: 6px !important;
    }
    .stRadio > label {
        color: #b0b8c1 !important;
    }
    div.stButton > button {
        background-color: #4b8bbe;
        color: white;
        font-weight: 700;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #3a6f9b;
        cursor: pointer;
    }
    input[type="range"] {
        accent-color: #4b8bbe;
    }
    .stSlider label {
        color: #b0b8c1;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 ChatGPT Dependency Predictor")

# Input fields
chatgpt_usage = st.number_input("ChatGPT usage frequency per week", 0, 50, 3)
duration = st.number_input("Average session duration (minutes)", 0, 300, 30)
attempts = st.number_input("Attempts before using ChatGPT", 0, 20, 2)
confidence = st.slider("Confidence in solving alone (1 = low, 5 = high)", 1, 5, 3)
peer_influence = st.slider("Peer usage influence (1 = none, 5 = strong)", 1, 5, 3)
reason = st.selectbox("Reason for using ChatGPT", ['No idea', 'Save time', 'Better answers'])
cgpa = st.number_input("CGPA (0.0 - 10.0)", 0.0, 10.0, 7.5, 0.01)
department = st.selectbox("Department", ['MECH', 'EXTC', 'COMPUTER', 'IT', 'ELECTRICAL', 'CIVIL'])
used_other_ai = st.radio("Used other AI tools?", [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
prefer_chatgpt = st.radio("Prefer ChatGPT over Google?", [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")

input_df = pd.DataFrame([{
    "chatgpt_usage_frequency_per_week": chatgpt_usage,
    "average_duration_per_session_minutes": duration,
    "attempt_before_chatgpt": attempts,
    "confidence_in_solving_alone": confidence,
    "reason_for_using_chatgpt": reason,
    "peer_usage_influence": peer_influence,
    "cgpa": cgpa,
    "department": department,
    "used_other_ai_tools": used_other_ai,
    "chatgpt_preferred_over_google": prefer_chatgpt
}])

if st.button("Predict Dependency"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown(f"### Prediction Confidence Score: {probability:.2f}")
    
    # Visual progress bar for dependency probability
    st.progress(int(probability * 100))

    # Color-coded message and advice
    if prediction == 1:
        st.error(f"🔍 You are predicted to be **dependent** on ChatGPT.")
        st.markdown("""
        #### Tips to reduce dependency and boost your skills:
        - 💡 Try solving problems for at least 30 minutes before asking ChatGPT.
        - 🧠 Use ChatGPT to verify your answers, not as the first step.
        - ✍️ Maintain a learning journal with key takeaways.
        - 🧑‍🤝‍🧑 Discuss with peers or professors before AI tools.
        - 📚 Refer to books, notes, or trusted videos before AI help.
        - ⏱️ Limit ChatGPT use to 30 minutes/day.
        - ❌ Avoid ChatGPT during assessments unless allowed.
        """)
        
        # Additional personalized tips based on inputs
        if confidence <= 2:
            st.info("🔎 Your low confidence in solving alone suggests practicing more without help.")
        if peer_influence >= 4:
            st.info("🤝 High peer influence detected — try building your own problem-solving style.")
        if chatgpt_usage >= 10:
            st.info("⌛ High ChatGPT usage — consider scheduling 'offline' study time.")
        
    else:
        st.success(f"✅ You are predicted to be **responsible** in your ChatGPT usage.")
        st.markdown("""
        🎉 Keep up your strong self-learning habits!
        - Consider mentoring peers.
        - Explore advanced topics independently.
        - Balance AI assistance and personal problem-solving.
        """)
        if confidence >= 4:
            st.balloons()
            st.info("💪 Your confidence is great! Keep challenging yourself.")
    
    # Show feature importance
    importances = model.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importances",
        labels={"Importance": "Importance Score", "Feature": "Feature"},
        text=importance_df["Importance"].apply(lambda x: f"{x:.3f}")
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
