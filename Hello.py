import streamlit as st
from langchain_helper import get_few_shot_db_chain

st.title("CarTalk AI: Unleashing Automotive Wisdom")

st.header("Question")
question = st.text_input(':')

if question:
    chain = get_few_shot_db_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)
