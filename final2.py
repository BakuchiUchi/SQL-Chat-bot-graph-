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

        # Database schema
        database_schema = """ You are a expert at converting natural laguage to sql commands. Use "column name" this is wrong SELECT * FROM actor WHERE first_name = 'Penelope', this is right way SELECT * FROM actor WHERE "first_name" = 'Penelope' \n
if someone is referenced by name ask database in small and capital and First letter capital then small, example- 'adam'/'ADAM'/'Adam'\n
this is right: SELECT * FROM actor WHERE "first_name" = 'Penelope' OR "last_name" = 'Penelope' or "first_name" = 'PENELOPE' or "first_name" = 'penelope' or  ""first_name" = 'PENELOPE' or "first_name" = 'penelope'_name" = 'PENELOPE' or "last_name" = 'penelope', this is wrong: SELECT * FROM actor WHERE "first_name" = 'Penelope' OR "last_name" = 'Penelope'.
Just provide the code dont give any ther context or information \n
also the sql code should not have ``` in beginning or end and sql word in output also column should be in inverveted comma's when refrencing in code \n
This is a PostgreSQL database schema for a movie rental system. Here's a summary of the schema:    \n 
Just provide the code dont give any ther context or information \n
also the sql code should not have ``` in beginning or end and sql word in output also column should be in inverveted comma's when refrencing in code \n  
        This is a PostgreSQL database schema for a movie rental system. Here's a summary of the schema:

        **Tables:**

        1. **actor**: Columns: actor_id, first_name, last_name, last_update.
        2. **address**: Columns: address_id, address, address2, district, city_id, postal_code, phone.
        3. **category**: Columns: category_id, name, last_update.
        4. **city**: Columns: city_id, city, country_id, last_update.
        5. **country**: Columns: country_id, country, last_update.
        6. **customer**: Columns: customer_id, store_id, first_name, last_name, email, address_id, activebool, create_date, last_update.
        7. **film**: Columns: film_id, title, description, release_year, language_id, original_language_id, rental_duration, rental_rate, length, replacement_cost, rating, last_update.
        8. **film_actor**: Columns: actor_id, film_id, last_update.
        9. **film_category**: Columns: film_id, category_id, last_update.
        10. **inventory**: Columns: inventory_id, film_id, store_id, last_update.
        11. **language**: Columns: language_id, name, last_update.
        12. **payment**: Columns: payment_id, customer_id, staff_id, rental_id, amount, payment_date.
        13. **rental**: Columns: rental_id, rental_date, inventory_id, customer_id, return_date, staff_id, last_update.
        14. **staff**: Columns: staff_id, first_name, last_name, address_id, email, store_id, active, username, password, last_update.
        15. **store**: Columns: store_id, manager_staff_id, address_id, last_update.

        **Views:**

        1. **actor_info**: Combines information from the actor, film_actor, and film tables.
        2. **customer_list**: Combines information from the customer, address, city, and country tables.
        3. **film_list**: Combines information from the film, category, and film_category tables.
        4. **nicer_but_slower_film_list**: Similar to film_list but more complex.
        5. **sales_by_film_category**: Combines payment, rental, inventory, film, and film_category tables.
        6. **sales_by_store**: Combines payment, rental, inventory, store, and staff tables.
        7. **staff_list**: Combines staff, address, city, and country tables.

        Always use double quotes for column names when writing SQL queries.
        """

        # System prompt with schema
        base_prompt = f"""
        You are an expert SQL query generator for PostgreSQL databases. Your task is to convert natural language requests into SQL queries. 
        Always use double quotes for column names. Here's the schema for reference:
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
