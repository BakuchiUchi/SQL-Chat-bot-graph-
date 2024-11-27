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

def display_chat_history(chat_history):
    """Display the chat history in the chat interface."""
    for chat in chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["content"])
        elif chat["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(chat["content"])

def plot_graph(df):
    """Generate a graph based on the DataFrame."""
    if len(df.columns) < 2:
        st.error("Not enough columns for a graph.")
        return
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 1:
        st.error("No numeric columns found for plotting.")
        return
    
    st.write("Generating graph...")
    fig, ax = plt.subplots()
    df.plot(kind="bar", x=df.columns[0], y=numeric_cols, ax=ax)
    st.pyplot(fig)

def main():
    # Groq API Key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Groq API key not found. Please check your environment variables.")
        return

    # Page layout setup
    st.set_page_config(
        page_title="NL2SQL Chatbot",
        layout="wide",
    )

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    display_chat_history(st.session_state.chat_history)

    # Hide the initial message after the first interaction
    if len(st.session_state.chat_history) == 0:
        st.title("NL2SQL Chatbot")
        st.write("Ask natural language questions about your database!")

    # Sidebar customization
    st.sidebar.title("Customization")
    system_prompt = st.sidebar.text_input("Additional system prompt:")
    model = st.sidebar.selectbox(
        "Choose a model",
        [ "llama-3.1-8b-instant", "gemma2-9b-it", "llama-3.1-70b-versatile"]
    )
    memory_length = st.sidebar.slider("Conversational memory length:", 1, 10, value=5)

    # Input box at the bottom of the screen
    with st.container():
        user_input = st.text_input(
            "Ask your question:",
            key="user_input",
            label_visibility="collapsed",
        )

    if user_input:
        conn = connect_to_db()
        if not conn:
            return

        # Define system prompt
        database_schema = """### Database Schema (PostgreSQL, Movie Rental System):  
#### **Tables**  
1. **actor**: actor_id (PK), first_name, last_name, last_update  
2. **address**: address_id (PK), address, address2, district, city_id (FK), postal_code, phone  
3. **category**: category_id (PK), name, last_update  
4. **city**: city_id (PK), city, country_id (FK), last_update  
5. **country**: country_id (PK), country, last_update  
6. **customer**: customer_id (PK), store_id (FK), first_name, last_name, email, address_id (FK), activebool, create_date, last_update  
7. **film**: film_id (PK), title, description, release_year, language_id (FK), original_language_id (FK), rental_duration, rental_rate, length, replacement_cost, rating, last_update  
8. **film_actor**: actor_id (FK), film_id (FK), last_update  
9. **film_category**: film_id (FK), category_id (FK), last_update  
10. **inventory**: inventory_id (PK), film_id (FK), store_id (FK), last_update  
11. **language**: language_id (PK), name, last_update  
12. **payment**: payment_id (PK), customer_id (FK), staff_id (FK), rental_id (FK), amount, payment_date  
13. **rental**: rental_id (PK), rental_date, inventory_id (FK), customer_id (FK), return_date, staff_id (FK), last_update  
14. **staff**: staff_id (PK), first_name, last_name, address_id (FK), email, store_id (FK), active, username, password, last_update  
15. **store**: store_id (PK), manager_staff_id (FK), address_id (FK), last_update  

#### **Views**  
1. **actor_info**: Combines actor, film_actor, and film tables.  
2. **customer_list**: Combines customer, address, city, and country tables.  
3. **film_list**: Combines film, category, and film_category tables.  
4. **sales_by_film_category**: Combines payment, rental, inventory, film, and film_category tables.  
5. **sales_by_store**: Combines payment, rental, inventory, store, and staff tables.  

#### **Functions**  
1. **film_in_stock**: Returns the number of copies of a movie in stock.  
2. **film_not_in_stock**: Returns the number of copies not in stock.  
3. **get_customer_balance**: Returns the balance of a customer’s account.  
4. **inventory_in_stock**: Checks if a particular inventory item is in stock.  
"""
        base_prompt = f"""You are an expert at converting natural language to SQL commands. Follow these rules:  
1. Use double quotes for column names (e.g., `SELECT * FROM actor WHERE "first_name" = 'Penelope'`).  
2. For name queries, search in lowercase, uppercase, and title case (e.g., `'adam'`, `'ADAM'`, `'Adam'`). Include variations for both "first_name" and "last_name".  
3. Only return the SQL code; do not include any extra context or formatting like ` ``` `.  
4. Ensure all column names are enclosed in double quotes in the output.
5. never give anything else but the sql querie and only give one query with any extra formatting ""  
Focus on generating accurate SQL queries using this schema, adhering to the outlined rules and best practices.\n
 Convert the user's question into an SQL query based on the schema below. Return only the query without explanations:
        {database_schema}"""
        if system_prompt:
            base_prompt += "\n" + system_prompt

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=base_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        llm = Groq(model="llama-3.1-70b-versatile", api_key=groq_api_key)
        groq_chat = ChatGroq(api_key=groq_api_key, model_name=model)

        conversation = LLMChain(
            llm=groq_chat, prompt=prompt, verbose=True
        )

        # Generate SQL query
        inputs = {
            "human_input": user_input,
            "chat_history": st.session_state.chat_history,  # Pass chat history here
        }
        sql_query = conversation.predict(**inputs)
        print(sql_query)

        # Execute the query
        df = execute_sql_query(conn, sql_query)

        # Append user input to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display the result
        if not df.empty:
            result_content = f"Query Results:\n{df.to_markdown()}"
            st.session_state.chat_history.append({"role": "assistant", "content": result_content})
            
            # Show graph button
            if st.button("Show Graph"):
                plot_graph(df)
        else:
            result_content = "No results found or query execution failed."
            st.session_state.chat_history.append({"role": "assistant", "content": result_content})

        # Refresh chat history
        display_chat_history(st.session_state.chat_history)

if __name__ == "__main__":
    main()
