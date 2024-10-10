# Databricks notebook source
# MAGIC %pip install langchain-huggingface
# MAGIC
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

from huggingface_hub import notebook_login

notebook_login()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### HuggingFace Langchain Integration

# COMMAND ----------

# DBTITLE 1,HuggingFace LangChain Integration
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

zephyr_llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

Phi3_llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

#Zephyr Chat
zephyr_chat = ChatHuggingFace(llm=zephyr_llm, verbose=True)

#Phi3 Chat
Phi3_chat = ChatHuggingFace(llm=Phi3_llm, verbose=True)

#Messages
messages = [
    ("system", "You are a helpful question answer bot. Answer the question asked by the user"),
    ("human", "What is the capital of France ? "),
]

# COMMAND ----------

zephyr_chat.invoke(messages)

# COMMAND ----------

Phi3_chat.invoke(messages)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Defining Tools

# COMMAND ----------

# DBTITLE 1,Search Tool
# MAGIC %pip install --upgrade --quiet  langchain-community langchain-databricks langchain-openai==0.1.19
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

from langchain_core.tools import Tool 
from langchain_databricks import ChatDatabricks
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import Agent, Tool
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# COMMAND ----------

import os
import pprint
import subprocess
import sys

%pip install langchain-community

from langchain_community.utilities import GoogleSerperAPIWrapper

#Define serper API key in environment variable 
os.environ["SERPER_API_KEY"] = "a6b875b80be0db48efb153468eadab3600252fa1"

search_tool = GoogleSerperAPIWrapper()
search_results = search_tool.run("what is the weather of SF?")
search_results

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP FUNCTION IF EXISTS daiwt_cafe_retail.ai.search_tool;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE FUNCTION daiwt_cafe_retail.ai.search_tool (
# MAGIC   query STRING COMMENT 'Query to execute the search results on'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'This functions searches the web based on user query'
# MAGIC AS $$
# MAGIC   import os
# MAGIC   import pprint
# MAGIC   import subprocess
# MAGIC   import sys
# MAGIC
# MAGIC   def install_and_import(package):
# MAGIC     try:
# MAGIC       __import__(package)
# MAGIC     except ImportError:
# MAGIC       subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# MAGIC
# MAGIC   # Example of installing and importing a package
# MAGIC   install_and_import('langchain-community')
# MAGIC
# MAGIC   from langchain_community.utilities import GoogleSerperAPIWrapper
# MAGIC
# MAGIC   #Define serper API key in environment variable 
# MAGIC   os.environ["SERPER_API_KEY"] = "a6b875b80be0db48efb153468eadab3600252fa1"
# MAGIC
# MAGIC   search_tool = GoogleSerperAPIWrapper()
# MAGIC   print(f'Search results for query: {query}')
# MAGIC   search_results = search_tool.run("what is the weather of SF?")
# MAGIC   return search_results
# MAGIC
# MAGIC $$

# COMMAND ----------

from langchain_openai import ChatOpenAI
from databricks.sdk import WorkspaceClient

# Note: langchain_community.chat_models.ChatDatabricks doesn't support create_tool_calling_agent yet - it'll soon be availableK. Let's use ChatOpenAI for now
llm = ChatOpenAI(
  base_url=f"{WorkspaceClient().config.host}/serving-endpoints/",
  api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
  model="databricks-meta-llama-3-70b-instruct"
)

# COMMAND ----------

def get_prompt(history = [], prompt = None):
    if not prompt:
            prompt = """You are a helful assistant. use search_tool to perform web search"""
    return ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
    ])
prompt = get_prompt()

# COMMAND ----------

search_tool = GoogleSerperAPIWrapper()
agent = create_tool_calling_agent(llm, search_tool, prompt)
agent_executor = AgentExecutor(agent=agent, tools=search_tool, verbose=True)

# COMMAND ----------



# COMMAND ----------


