from langchain_experimental.agents import create_pandas_dataframe_agent
#from langchain.agents.agent_toolkits import create_pandas_dataframe_tool
from langchain.agents import initialize_agent, AgentType
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI

def query_agent(data, query):

    # Parse the CSV file and create a Pandas DataFrame from its contents.
    df = pd.read_csv(data)

    llm = GoogleGenerativeAI(model="gemini-2.0-flash")

     # Create a Pandas tool from the DataFrame
    #pandas_tool = create_pandas_dataframe_tool(df)
    
    
      # Initialize agent with Gemini and Python execution allowed
    agent = create_pandas_dataframe_agent(
       # tools=[pandas_tool],
        llm=llm,
        df=df,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ✅ Gemini-friendly agent type
        verbose=True,
        allow_dangerous_code=True                     # ✅ Enables Python REPL execution
    )


    #Python REPL: A Python shell used to evaluating and executing Python commands. 
    #It takes python code as input and outputs the result. The input python code can be generated from another tool in the LangChain
    return agent.run(query)