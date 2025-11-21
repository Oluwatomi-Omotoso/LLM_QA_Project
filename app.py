import streamlit as st
import google.generativeai as genai
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NLP Q&A System", page_icon="🤖")

# --- API SETUP ---
# Ideally, use st.secrets for deployment, but for this submission format:
# You can hardcode the key here OR use a sidebar input for safety.
# If you hardcode it, ensure you don't expose it publicly if possible.
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")


def preprocess_input(user_text):
    """
    Applies lowercasing, punctuation removal, and tokenization.
    """
    text = user_text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return " ".join(tokens), tokens


# --- UI LAYOUT ---
st.title("🤖 NLP Q&A System")
st.subheader("Powered by Google Gemini")

# Input Area
user_question = st.text_area("Enter your question here:", height=100)

if st.button("Get Answer"):
    if not api_key:
        st.error("Please enter an API Key in the sidebar to proceed.")
    elif not user_question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing and asking the AI..."):
            try:
                # Configure API with the key provided by user
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-flash-lite")

                # Preprocessing
                processed_text, tokens = preprocess_input(user_question)

                # API Call
                response = model.generate_content(processed_text)

                # --- DISPLAY RESULTS ---
                st.divider()

                # 1. View Processed Question (Requirement)
                col1, col2 = st.columns(2)
                with col1:
                    st.info("Processed Text (Lower/Cleaned)")
                    st.write(processed_text)
                with col2:
                    st.info("Tokens Generated")
                    st.write(tokens)

                # 2. Display Generated Answer (Requirement)
                st.success("LLM API Response")
                st.markdown(response.text)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("CSC331 Project 2 | NLP Q&A System")
