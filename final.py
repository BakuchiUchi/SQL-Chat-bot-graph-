import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
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
            results = cur.fetchall()
        conn.commit()
        return results
    except Exception as e:
        st.error("Error executing the SQL query: " + str(e))
        return []
    finally:
        conn.close()

def generate_response(llm, question, system_prompt):
    """Generate a response using the LLM."""
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=question),
    ]
    try:
        response = llm.chat(messages)
        return str(response).strip()
    except Exception as e:
        st.error("Error generating response: " + str(e))
        return ""

def main():
    # Groq API Key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Groq API key not found. Please check your environment variables.")
        return

    # Initialize Streamlit app
    st.title("NL2SQL Chatbot")
    st.write("Convert natural language to SQL queries effortlessly!")

    # Sidebar customization
    st.sidebar.title("Additional Customization")
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

    # User input
    user_question = st.text_input("Ask a question:")

    # Initialize Groq chat
    llm = Groq(model="llama-3.1-70b-versatile", api_key=groq_api_key)
    groq_chat = ChatGroq(api_key=groq_api_key, model_name=model)

    # Restore session history in memory
    for msg in st.session_state.chat_history:
        memory.save_context({"input": msg["human"]}, {"output": msg["AI"]})

    if user_question:
        conn = connect_to_db()
        if not conn:
            return

        # Define the SQL-focused system prompt
        base_prompt = f"""
        You are an expert SQL query generator for PostgreSQL databases. Your task is to convert natural language requests into SQL queries. 
        Always use double quotes for column names. Here's an example:
        
        Request: "List all actors named 'Penelope'."
        Response: SELECT * FROM "actor" WHERE "first_name" = 'Penelope';
        
        Do not add explanations. Only return the SQL query.
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
        sql_query = conversation.predict(human_input=user_question)
        #st.write("Generated SQL Query:", sql_query)

        # Execute SQL and process results
        results = execute_sql_query(conn, sql_query)
        if results:
            st.write("Query Results:")
            st.write(results)

            # Update chat history
            st.session_state.chat_history.append({"human": user_question, "AI": sql_query})
        else:
            st.write("No results or an error occurred while executing the SQL query.")

if __name__ == "__main__":
    main()
