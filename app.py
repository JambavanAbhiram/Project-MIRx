import streamlit as st

from rag_pipeline import run_rag

st.set_page_config(
    page_title="MIRx AI",
)

st.title("MIRx AI")

query = st.text_input(
    "Ask your medical question"
)

if st.button("Generate"):

    if query.strip() != "":

        with st.spinner("Generating response..."):

            response = run_rag(query)

            st.subheader("Response")

            st.write(response)