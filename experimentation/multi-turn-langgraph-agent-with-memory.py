# Databricks notebook source
# MAGIC %pip install --upgrade --quiet databricks-sdk langchain-community mlflow langchain-openAI langgraph
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

# COMMAND ----------

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")

tools = (UCFunctionToolkit( warehouse_id=wh.id).include("mosaic_agent.agent.*",).get_tools())


model = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    print(f"the message is : {messages}")
    last_message = messages[-1]

    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END

# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)

    return {"messages": [response]}

# Define a new LangGraph Object
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`. This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Now we compile it, This compiles it into a LangChain Runnable,
# meaning it can be used like any other runnable.Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# COMMAND ----------

from IPython.display import Image, display
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)

# COMMAND ----------

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the status of my order ORD907?")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content

# COMMAND ----------

b# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="what did i Order? I Forgot")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content

# COMMAND ----------

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="Can you compare the price of that product within your company vs current market?")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content
