# Databricks notebook source
# MAGIC %pip install --upgrade --quiet databricks-sdk langchain-community mlflow langchain-openAI
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------


from langchain_openai import ChatOpenAI
from databricks.sdk import WorkspaceClient

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

# COMMAND ----------

from langchain_community.tools.databricks import UCFunctionToolkit
import pandas as pd

def display_tools(tools):
    display(pd.DataFrame([{k: str(v) for k, v in vars(tool).items()} for tool in tools]))

tools = (
    UCFunctionToolkit(
        # You can find the SQL warehouse ID in its UI after creation.
        warehouse_id=wh.id
    )
    .include(
        # Include functions as tools using their qualified names.
        # You can use "{catalog_name}.{schema_name}.*" to get all functions in a schema.
        "mosaic_agent.agent.*"
    )
    .get_tools()
)

display_tools(tools)

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks

# Note: langchain_community.chat_models.ChatDatabricks doesn't support create_tool_calling_agent 
# yet - it'll soon be availableK. Let's use ChatOpenAI for now
# Create the llm via ChatDatabricks 

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")


# llm = ChatOpenAI(
#   base_url=f"{WorkspaceClient().config.host}/serving-endpoints/",
#   api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
#   model="databricks-meta-llama-3-70b-instruct"
# )



# COMMAND ----------

# Define the prompt for the model , note the description to use the tools
 
prompt = ChatPromptTemplate.from_messages(
    [(
        "system",
        "You are a helpful assistant for a large retail company.Make sure to use tool for information.Refer the tools description and make a decision of the tools to call for each user query.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# COMMAND ----------

agent_executor.invoke({"input": "Whats the price of breville electric kettle  in your company and in market?"})
