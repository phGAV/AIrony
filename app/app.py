import streamlit as st
import httpx
import os
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AIrony", layout="wide")

st.title("AIrony")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Input section
topic = st.text_input("Enter your meme topic:")
style = st.selectbox("Choose style:", ["Funny", "Sarcastic", "Wholesome"])

if st.button("Generate Meme"):
    if topic:
        with st.spinner("Generating your meme..."):
            try:
                # Call backend API
                logger.info('calling backend api')
                response = httpx.post(
                    f"{BACKEND_URL}/generate_meme",
                    json={"topic": topic, "style": style},
                    timeout=60.0
                )
                response.raise_for_status()
                meme_data = response.json()
                logger.info(f'received data from backend: {meme_data}')
                # Display meme with error handling
                try:
                    image_response = httpx.get(meme_data["url"], params={"font": "impact"})
                    if image_response.status_code == 200:
                        st.image(image_response.content)
                    else:
                        st.error(f"Failed to load image: {response.json()}")
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                
                # Display regenerate button
                if st.button("Regenerate"):
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error generating meme: {str(e)}")
    else:
        st.warning("Please enter a topic")