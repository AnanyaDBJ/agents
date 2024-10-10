# Databricks notebook source
# MAGIC %pip install -U langgraph langchain langchain_experimental langsmith databricks-sdk langchain-community mlflow langchain-openAI 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_openai import ChatOpenAI
from databricks.sdk import WorkspaceClient
from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_community.tools.databricks import UCFunctionToolkit
import mlflow

mlflow.langchain.autolog(disable=False)

def get_shared_warehouse(name=None):
    w = WorkspaceClient()
    warehouses = w.warehouses.list()
    for wh in warehouses:
        if wh.name == name:
            return wh
    for wh in warehouses:
        if wh.name.lower() == "shared endpoint":
            return wh
    for wh in warehouses:
        if wh.name.lower() == "dbdemos-shared-endpoint":
            return wh
    #Try to fallback to an existing shared endpoint.
    for wh in warehouses:
        if "dbdemos" in wh.name.lower():
            return wh
    for wh in warehouses:
        if "shared" in wh.name.lower():
            return wh
    for wh in warehouses:
        if wh.num_clusters > 0:
            return wh       
    raise Exception("Couldn't find any Warehouse to use. Please create a wh first to run the demo and add the id here")

wh = get_shared_warehouse(name = None) 
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")

tools = (UCFunctionToolkit( warehouse_id=wh.id).include("mosaic_agent.agent.*",).get_tools())

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [(
        "system",
        "You are a helpful assistant. Make sure to use tool for information.Just return the response , do not need to mention about function call.Utilize appropriate judgement for routing user question to specific tools.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "Whats the price of breville electric kettle  in your company and in market?"})

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Multi-turn conversational agent

# COMMAND ----------

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
memory = ChatMessageHistory(session_id="sample-validation")

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# COMMAND ----------

agent_with_chat_history.invoke(
    {"input": "What is the status of my order ORD907?"},
    config={"configurable": {"session_id": "sess01"}},
)

# COMMAND ----------

agent_with_chat_history.invoke(
    {"input": "What is the status of the shipment of my order ?"},
    config={"configurable": {"session_id": "sess01"}},
)

# COMMAND ----------

agent_with_chat_history.invoke(
    {"input": "What type of product did i order ? I forgot"},
    config={"configurable": {"session_id": "sess01"}},
)
