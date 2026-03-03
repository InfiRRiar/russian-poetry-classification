import streamlit as st
import httpx

BACKEND_URL = "http://127.0.0.1:8000/process"

st.title("Poem classificator")

user_text = st.text_area("Enter your poem here:")

if st.button("Send"):
    if not user_text.strip():
        st.warning("Text is empty")
    else:
        try:
            response = httpx.post(
                BACKEND_URL,
                json={"text": user_text},
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                st.success("Response received")
                st.write("Received text:", data["received_text"])
                st.write("Length:", data["length"])
            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Request failed: {e}")