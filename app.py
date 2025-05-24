import streamlit as st
import pandas as pd
import pickle

# Load the trained pipeline
with open("chatgpt_model.pkl", "rb") as f:
    model = pickle.load(f)

# Custom CSS for classy dark theme with blue accents
st.markdown(
    """
    <style>
    /* Background and font */
    .reportview-container, .main {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Title style */
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

    input[type=number] {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
        border: 1px solid #3a3f51 !important;
        border-radius: 4px !important;
        padding: 6px !important;
    }

    div[role="listbox"] {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
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

# Title
st.title("🤖 ChatGPT Dependency Predictor")

# Inputs
chatgpt_usage = st.number_input(
    "ChatGPT usage frequency per week", min_value=0, max_value=50, value=3, step=1
)
duration = st.number_input(
    "Average session duration (minutes)", min_value=0, max_value=300, value=30, step=1
)
attempts = st.number_input(
    "Attempts before using ChatGPT", min_value=0, max_value=20, value=2, step=1
)
confidence = st.slider(
    "Confidence in solving alone (1 = low, 5 = high)", 1, 5, 3
)
peer_influence = st.slider(
    "Peer usage influence (1 = none, 5 = strong)", 1, 5, 3
)
reason = st.selectbox(
    "Reason for using ChatGPT", ['No idea', 'Save time', 'Better answers']
)
cgpa = st.number_input(
    "CGPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, step=0.01, value=7.5
)
department = st.selectbox(
    "Department", ['MECH', 'EXTC', 'COMPUTER', 'IT', 'ELECTRICAL', 'CIVIL']
)
used_other_ai = st.radio(
    "Used other AI tools?", [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes"
)
prefer_chatgpt = st.radio(
    "Prefer ChatGPT over Google?", [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes"
)

# Prepare input
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

# Predict
if st.button("Predict Dependency"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🔍 Predicted ChatGPT Dependency: Yes")
        st.write(f"Confidence score: {probability:.2f}")
        st.markdown("### 🧠 Tips to Reduce ChatGPT Dependency and Boost Critical Thinking:")
        st.markdown("""
        - 💡 **Try solving problems on your own** for at least 30 minutes before asking ChatGPT.
        - 🧠 **Use ChatGPT to verify** your answers, not as the first step.
        - ✍️ **Maintain a learning journal** where you write key takeaways instead of copy-pasting.
        - 🧑‍🤝‍🧑 **Discuss with peers or professors** before turning to AI tools.
        - 📚 **Refer to books, lecture notes, or trusted educational videos** before seeking AI help.
        - ⏱️ **Limit usage duration** (e.g., no more than 30 minutes/day).
        - ❌ Avoid using ChatGPT during assessments unless explicitly allowed.
        """)
    else:
        st.success(f"✅ Predicted ChatGPT Dependency: No")
        st.write(f"Confidence score: {probability:.2f}")
        st.markdown("### 🎉 You're using ChatGPT responsibly!")
        st.markdown("Keep up your strong self-learning habits. Try mentoring peers or exploring advanced topics independently. 🚀")
