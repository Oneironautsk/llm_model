import streamlit as st
from langchain_helper import get_few_shot_db_chain

st.title("LLM-Query Hub : Effortless Data Retrieval And Analysis\n")

st.write(' ')

question = st.text_input('## Ask about features, specs, or anything about cars')

if question:
    chain = get_few_shot_db_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)
