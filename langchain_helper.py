from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain.prompts.prompt import PromptTemplate
import streamlit as st
import subprocess
command = [
    "curl",
    "--create-dirs",
    "-o",
    f"{subprocess.os.environ['HOME']}/.postgresql/root.crt",
    "https://cockroachlabs.cloud/clusters/6b36bc32-0a92-42ba-96f6-08ae82326ebe/cert"
]
result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)

few_shots = [
    {'Question' : "what is the price of Toyota Fortuner car?",
     'SQLQuery' : "SELECT average_price FROM cardetails WHERE car_name = 'Toyota Fortuner'",
     'SQLResult': "Result of the SQL query",
     'Answer' : "4243500.0"},
    {'Question': "how many seats are in Audi q8?",
     'SQLQuery':"SELECT description FROM cardetails WHERE car_name = 'Audi Q8'",
     'SQLResult': "Result of the SQL query",
     'Answer': "5"},
    {'Question': "suggest me an best bmw car",
     'SQLQuery':"SELECT car_name FROM cardetails WHERE car_name LIKE '%BMW%' ORDER BY average_price,average_mileage limit 1",
     'SQLResult': "Result of the SQL query",
     'Answer': "BMW X1"},
    {'Question': "why audi q8 is a best car?",
     'SQLQuery':"SELECT car_name, average_price, average_mileage, engine_cc FROM cardetails WHERE car_name = 'Audi Q8'",
     'SQLResult': "Result of the SQL query",
     'Answer': "Audi Q8 is a best car because it has a high average mileage of 19.25 and a high engine cc of 2995 cc."},
]


def get_few_shot_db_chain():
    database_url = "cockroachdb://jeevanathan:DylCAmQbz0Y3UqpXPIlDzg@spirit-python-7118.8nk.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"
    google_api_key = st.secrets['google_api_key']

    db = SQLDatabase.from_uri(database_url,sample_rows_in_table_info=3)
    llm =  GooglePalm(google_api_key="AIzaSyC0l3tfYp4fDF4AQG9aDqDVHk9rMFJaxg8",temperature=0.1)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = FAISS.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
        Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
        Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
        Pay attention to use only the column names you can see in the tables info below. Be careful to not query for columns that do not exist. for the price use average price column and use starting and ending price if they needed.For the
        car brand name use first word from car_name column.To suggest a best car you have look into column like average mileage, average price and engine_cc.
        Use the following format:

        Question: Question here
        SQLQuery: Query to run with no pre-amble
        SQLResult: Result of the SQLQuery
        Answer: Final answer here

        No pre-amble.
        """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],
    )
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain
