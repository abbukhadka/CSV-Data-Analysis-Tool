# CSV Data Analysis Tool 

This project is a user-friendly web application that empowers anyone to explore and analyze CSV datasets through natural language queries. Built with **Streamlit** and powered by the advanced **Google Gemini 2.0** model via **LangChain**, this tool transforms how you interact with your data.

---

## Features

- **Simple CSV Upload:** Drag and drop or select your CSV file with ease.  
- **Conversational Queries:** Ask complex questions in plain English about your data.  
- **Smart AI Agent:** Utilizes the latest Google Gemini 2.0 LLM integrated with LangChain’s Pandas DataFrame agent for intelligent data understanding.  
- **Dynamic Python Execution:** Automatically runs Python code behind the scenes to analyze data and generate accurate responses.  
- **Instant Results:** Get fast, clear answers displayed in an interactive Streamlit interface.

---

## Technologies Used

- **Streamlit:** Rapidly build interactive data apps with a sleek UI.  
- **LangChain:** Coordinate AI-powered interactions with data and code execution.  
- **Google Gemini 2.0:** Next-gen large language model for natural language understanding and code generation.  
- **Pandas:** Reliable data handling and manipulation library.  
- **Python-dotenv:** For managing environment variables securely.

---

## How It Works

- **The user uploads a CSV file** 
- **The file is parsed into a Pandas DataFrame** 
- **The LangChain Pandas DataFrame agent, powered by Gemini 2.0, interprets the user's query**  
- **The agent generates and executes Python code dynamically to answer the query** 
- **The answer is displayed on the Streamlit interface in a user-friendly manner** 

