import streamlit as st
import joblib
import numpy as np

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Mental Burnout Predictor",
    page_icon="🧠",
    layout="wide"
)

# =========================
# LOAD MODEL & SCALER
# =========================

model = joblib.load("burnout_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# TITLE
# =========================

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>
    🧠 Student Mental Burnout Risk Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align: center;'>
    AI-Based Psychological Burnout Assessment System
    </h4>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
    This AI system predicts mental burnout risk among students using Machine Learning techniques.

    Developed using:
    - Python
    - Scikit-learn
    - Streamlit
    """
)

# =========================
# INSTRUCTIONS
# =========================

st.info(
    """
    Please answer all questions honestly based on your recent experiences.

    Scale:
    1 = Never  
    2 = Occasionally  
    3 = Sometimes  
    4 = Often  
    5 = Always
    """
)

# =========================
# QUESTIONS
# =========================

q1 = st.slider(
    "1. I feel mentally exhausted after studying or attending classes.",
    1, 5
)

q2 = st.slider(
    "2. I feel emotionally drained because of academic responsibilities.",
    1, 5
)

q3 = st.slider(
    "3. Even after resting, I still feel mentally tired.",
    1, 5
)

q4 = st.slider(
    "4. I struggle to regain energy after a stressful academic day.",
    1, 5
)

q5 = st.slider(
    "5. I feel constant pressure to perform well academically.",
    1, 5
)

q6 = st.slider(
    "6. I worry excessively about grades, assignments, or exams.",
    1, 5
)

q7 = st.slider(
    "7. I feel that my academic workload is too heavy to manage comfortably.",
    1, 5
)

q8 = st.slider(
    "8. My sleep schedule has become unhealthy because of academic stress.",
    1, 5
)

q9 = st.slider(
    "9. I wake up feeling tired or unrefreshed.",
    1, 5
)

q10 = st.slider(
    "10. I experience physical exhaustion due to studying or academic pressure.",
    1, 5
)

q11 = st.slider(
    "11. I sacrifice sleep or rest to complete academic tasks.",
    1, 5
)

q12 = st.slider(
    "12. I find it difficult to stay motivated toward my studies.",
    1, 5
)

q13 = st.slider(
    "13. I procrastinate even when I know tasks are important.",
    1, 5
)

q14 = st.slider(
    "14. I feel less interested in learning than I used to.",
    1, 5
)

q15 = st.slider(
    "15. I struggle to concentrate during studying or lectures.",
    1, 5
)

q16 = st.slider(
    "16. I become irritated or frustrated more easily than before.",
    1, 5
)

q17 = st.slider(
    "17. I feel emotionally disconnected from my studies or responsibilities.",
    1, 5
)

q18 = st.slider(
    "18. I feel hopeless or discouraged about managing my academic life.",
    1, 5
)

q19 = st.slider(
    "19. I feel emotionally pressured by expectations from myself.",
    1, 5
)

q20 = st.slider(
    "20. Academic stress negatively affects my personal life, relationships, or social interactions.",
    1, 5
)

# =========================
# PREDICTION BUTTON
# =========================

if st.button("🔍 Predict Burnout Risk"):

    with st.spinner("Analyzing mental burnout patterns..."):

        input_data = np.array([[
            q1, q2, q3, q4, q5,
            q6, q7, q8, q9, q10,
            q11, q12, q13, q14, q15,
            q16, q17, q18, q19, q20
        ]])

        # Scale input
        scaled_data = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(scaled_data)

        # Total score
        total_score = np.sum(input_data)

        st.divider()

        st.subheader("📊 Prediction Result")

        st.write("### Total Burnout Score:", total_score)

        # =========================
        # RESULT DISPLAY
        # =========================

        if prediction[0] == "Low":

            st.success("🟢 Low Burnout Risk")

            st.write(
                """
                ✅ Maintain healthy study habits  
                ✅ Continue proper sleep schedule  
                ✅ Take regular breaks  
                """
            )

        elif prediction[0] == "Moderate":

            st.warning("🟡 Moderate Burnout Risk")

            st.write(
                """
                ⚠️ Manage academic stress carefully  
                ⚠️ Improve sleep quality  
                ⚠️ Reduce overload and overthinking  
                """
            )

        else:

            st.error("🔴 High Burnout Risk")

            st.write(
                """
                🚨 Strong burnout indicators detected  
                🚨 Consider stress management strategies  
                🚨 Prioritize mental and physical recovery  
                """
            )

# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "AI-Based Mental Burnout Prediction System | BSCS AI Project"
)