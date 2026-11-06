import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "asd-backend"))
from asd_core import get_supabase, load_model, predict, save_prediction  # noqa: E402

# AQ-10 items. A point is scored for "agree" on items 1, 7, 8, 10 and for
# "disagree" on items 2, 3, 4, 5, 6, 9.
QUESTIONS = [
    ("A1", "I often notice small sounds when others do not", "agree"),
    ("A2", "I usually concentrate more on the whole picture, rather than the small details", "disagree"),
    ("A3", "I find it easy to do more than one thing at once", "disagree"),
    ("A4", "If there is an interruption, I can switch back to what I was doing very quickly", "disagree"),
    ("A5", "I find it easy to ‘read between the lines’ when someone is talking to me", "disagree"),
    ("A6", "I know how to tell if someone listening to me is getting bored", "disagree"),
    ("A7", "When I’m reading a story, I find it difficult to work out the characters’ intentions", "agree"),
    ("A8", "I like to collect information about categories of things", "agree"),
    ("A9", "I find it easy to work out what someone is thinking or feeling just by looking at their face", "disagree"),
    ("A10", "I find it difficult to work out people’s intentions", "agree"),
]


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_db():
    # Enabled when SUPABASE_URL and SUPABASE_KEY are set (Space secrets on Hugging Face).
    return get_supabase()


def show_predict_page():
    st.title("ASD Detection System")
    st.write("Answer each statement, then press Predict.")

    with st.form("user_data"):
        answers = {}
        for i, (key, text, scored) in enumerate(QUESTIONS, 1):
            choice = st.radio(f"{i}. {text}", ["Agree", "Disagree"], horizontal=True, key=key)
            answers[key] = int(choice.lower() == scored)
        age = st.number_input("Age", min_value=1, max_value=100, step=1, value=18)
        sex = st.selectbox("Sex", ["m", "f"], format_func={"m": "Male", "f": "Female"}.get)
        jaundice = st.selectbox("Born with jaundice?", ["no", "yes"])
        family_asd = st.selectbox("Family history of ASD?", ["no", "yes"])
        ok = st.form_submit_button("Predict")

    if ok:
        payload = {**answers, "Age": age, "Sex": sex, "Jaundice": jaundice, "Family_ASD": family_asd}
        label, proba = predict(get_model(), payload)
        db = get_db()
        save_error = None
        if db is not None:
            try:
                save_prediction(db, payload, label)
            except Exception as e:
                # A database outage should not block the screening result.
                save_error = e
                print(f"Failed to store prediction in Supabase: {e}")
        elif os.environ.get("SUPABASE_URL"):
            save_error = "Supabase client could not start (check SUPABASE_KEY)"
        if label == 1:
            st.warning(f"Prediction: ASD traits detected (probability {proba:.0%})")
        else:
            st.success(f"Prediction: No ASD traits detected (probability {proba:.0%})")
        st.caption("This is a screening aid, not a diagnosis. Please consult a qualified clinician.")
        if save_error is not None:
            st.caption(f"Note: this result was not saved ({save_error}).")
