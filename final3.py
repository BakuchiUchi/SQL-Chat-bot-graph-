import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from llama_index.llms.groq import Groq
import psycopg2

# Load environment variables
load_dotenv()

def connect_to_db():
    """Establish and return a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            database="numbertwo",
            host="localhost",
            user="postgres",
            password="admin",
            port="5432"
        )
        return conn
    except psycopg2.Error as e:
        st.error("Error connecting to the database: " + str(e))
        return None

def execute_sql_query(conn, sql_query):
    """Execute an SQL query and fetch results."""
    try:
        with conn.cursor() as cur:
            cur.execute(sql_query)
            colnames = [desc[0] for desc in cur.description]  # Get column names
            results = cur.fetchall()  # Fetch all rows
        conn.commit()
        return pd.DataFrame(results, columns=colnames)
    except Exception as e:
        st.error("Error executing the SQL query: " + str(e))
        return pd.DataFrame()  # Return an empty DataFrame on failure
    finally:
        conn.close()

def plot_graph(dataframe):
    """Plot a graph if the dataframe contains numerical data."""
    try:
        if dataframe.shape[1] >= 2 and dataframe.dtypes[1].kind in 'if':  # Check if second column is numeric
            st.line_chart(dataframe)
        else:
            st.error("Data is not suitable for graphing.")
    except Exception as e:
        st.error("Error generating graph: " + str(e))

def main():
    # Groq API Key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Groq API key not found. Please check your environment variables.")
        return

    # Initialize Streamlit app
    st.title("NL2SQL Chatbot")
    st.write("Ask natural language questions about your database!")

    # Sidebar customization
    st.sidebar.title("Customization")
    system_prompt = st.sidebar.text_input("Additional system prompt:")
    model = st.sidebar.selectbox(
        "Choose a model",
        ["llama-3.1-70b-specdec", "llama-3.1-8b-instant", "gemma2-9b-it", "llama-3.1-70b-versatile"]
    )
    memory_length = st.sidebar.slider("Conversational memory length:", 1, 10, value=5)

    # Conversational memory
    memory = ConversationBufferWindowMemory(
        k=memory_length, memory_key="chat_history", return_messages=True
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize Groq chat
    llm = Groq(model="llama-3.1-70b-versatile", api_key=groq_api_key)
    groq_chat = ChatGroq(api_key=groq_api_key, model_name=model)

    # Restore session history in memory
    for msg in st.session_state.chat_history:
        memory.save_context({"input": msg["human"]}, {"output": msg["AI"]})

    # Chat interface
    user_input = st.text_input("Ask your question:", key="user_input")

    if user_input:
        conn = connect_to_db()
        if not conn:
            return

        # Define system prompt
        database_schema = """ 
        You are an SQL expert assisting with a PostgreSQL database. Here is the schema:
        1. actor: actor_id, first_name, last_name, last_update
        2. address: address_id, address, address2, district, city_id, postal_code, phone
        3. film: film_id, title, description, release_year, language_id, rental_rate, rating, last_update
        4. customer: customer_id, store_id, first_name, last_name, email, address_id, create_date, last_update
        """
        base_prompt = f""" You are a expert at converting natural laguage to sql commands. Use "column name" this is wrong SELECT * FROM actor WHERE first_name = 'Penelope', this is right way SELECT * FROM actor WHERE "first_name" = 'Penelope' \n
if someone is referenced by name ask database in small and capital and First letter capital then small, example- 'adam'/'ADAM'/'Adam'\n
this is right: SELECT * FROM actor WHERE "first_name" = 'Penelope' OR "last_name" = 'Penelope' or "first_name" = 'PENELOPE' or "first_name" = 'penelope' or  ""first_name" = 'PENELOPE' or "first_name" = 'penelope'_name" = 'PENELOPE' or "last_name" = 'penelope', this is wrong: SELECT * FROM actor WHERE "first_name" = 'Penelope' OR "last_name" = 'Penelope'.
Just provide the code dont give any ther context or information \n
also the sql code should not have ``` in beginning or end and sql word in output also column should be in inverveted comma's when refrencing in code \n
This is a PostgreSQL database schema for a movie rental system. Here's a summary of the schema:    \n 
Just provide the code dont give any ther context or information \n
also the sql code should not have ``` in beginning or end and sql word in output also column should be in inverveted comma's when refrencing in code \n
        Convert the user's question into an SQL query based on the schema below. Return only the query without explanations:
        {database_schema}
        """
        if system_prompt:
            base_prompt += "\n" + system_prompt

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=base_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        conversation = LLMChain(
            llm=groq_chat, prompt=prompt, verbose=True, memory=memory
        )

        # Generate SQL query
        sql_query = conversation.predict(human_input=user_input)

        # Execute the query
        df = execute_sql_query(conn, sql_query)

        # Display the result
        with st.chat_message("user"):
            st.write(user_input)

        if not df.empty:
            with st.chat_message("assistant"):
                st.write("Query Results:")
                st.dataframe(df)

                # Generate a graph if applicable
                st.write("Visualizing Results:")
                plot_graph(df)
        else:
            with st.chat_message("assistant"):
                st.write("No results found or query execution failed.")

        # Update chat history
        st.session_state.chat_history.append({"human": user_input, "AI": "Results displayed."})

if __name__ == "__main__":
    main()
