import streamlit as st
from langchain_helper import get_few_shot_db_chain

st.title("DECISION SUPPORT SYSTEM FOR CAR BUYERS\n")
st.write(' ')
st.write(' ')
st.write('#### Ask about features, specs, or anything about cars :')
question = st.text_input('')

if question:
    chain = get_few_shot_db_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)
