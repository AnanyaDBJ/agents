# Databricks notebook source
# MAGIC %pip install -U langgraph==0.2.57 langchain_community  langchain_experimental databricks-sdk  databricks-langchain mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------


from databricks.sdk import WorkspaceClient
from langchain_community.tools.databricks import UCFunctionToolkit
import mlflow

mlflow.langchain.autolog()

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

#Get the latest warehouse ID name
wh = get_shared_warehouse(name = None) 
#Extract all the tools present in UC
uc_tools = UCFunctionToolkit(warehouse_id=wh.id).include("mosaic_agent.online_electric_retailer.*").get_tools()

#Print all the tools
for element in uc_tools:
  print(element.name)

# COMMAND ----------

from langgraph.prebuilt import create_react_agent
from databricks_langchain import ChatDatabricks
import mlflow

mlflow.langchain.autolog()

# llm = ChatDatabricks(endpoint="agents-demo-gpt4o")
# llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
llm = ChatDatabricks(endpoint="gpt-4o-mini_yyl")

# COMMAND ----------

from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph import MessagesState
from langgraph.types import Command


members = ["Order","Complaint","Product"]

# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed

options = members + ["FINISH"]

system_prompt = (
    """You are a strategic supervisor workflow coordinating between : {members}. Your role is to :
        1. Analyze the current conversation state and task progress
        2. Determine the most appropriate worker for the next step
        3. Ensure efficient task completion without redundant work
        4. Monitor overall conversation quality and coherence

        You might need to perform multi-stage iteration to get an answer. 

        For each worker:
        - Order: Use this agent to retrive and analyse questions that are related to Order and Shipment. You can also use it to extract necessary product or shipment ID details to answer other questions.It can be :
          1. Query on status of Order 
          2. Query on Product or Shipment and get the relevant ID from the response
          2. Extract relevant incident number 

        - Complaint: Use this agent to retrieve and analyze Incident and Complaint information providing insights into customer complaint and incident reasons. 

        - Product: Use this agent to retrive and analyse questions that are related to Product and provide product recommendation. It can be:
        1. Query based on product ID and provide product details 
        2. Recommend relevant products suggestions to customer based on their requirement

        Only respond with FINISH when:
        - The user's question has been fully answered
        - All necessary information has been gathered and processed
        - The response quality meets high standards
    """
)
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]

def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    print(f'next state : {goto}')
    if goto == "FINISH":
        goto = END

    return Command(goto=goto)

# COMMAND ----------

from langchain_core.messages import HumanMessage,AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

order_agent = create_react_agent(
    llm, 
    tools=[uc_tools[5]], 
    state_modifier="You are a Order management agent who does everything on Order. From Order Query , to Order Placement , Shipment Status enquiry and Order Return. Do not perfrom any other task. Once you have received the answer , return the result in a structured format , Append the word 'Successfully completed' at the start of the message and write response in one sentence."
)

def order_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = order_agent.invoke(state)
    dummy_response = result["messages"][-1].content
    return Command(
        update={
            "messages": [
                AIMessage(content=dummy_response, name="Order")

            ]
        },
        goto="supervisor",
    )

model = llm.bind_tools(uc_tools)

complaint_agent = create_react_agent(
    llm, 
    tools=[uc_tools[3]], 
    state_modifier="You are a Incident/Complaint management agent who does everything on Incident. From Incident Query , to new Incident Logging to Incident Resolution. Do not perform any other task.Return the result in a structured format , Append the word 'Successfully completed' at the start of the message. "
)

def complaint_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    # result = complaint_agent.invoke(state)
    dummy_response = "Successfully completed:The status of the Complaint shows Pending."
    return Command(
        update={
            "messages": [
                HumanMessage(content=dummy_response, name="Complaint")
            ]
        },
        goto="supervisor",
    )

product_agent = create_react_agent(
    llm, 
    tools=[uc_tools[6],uc_tools[1]], 
    state_modifier="You are an Agent only for Product Management & Recommendation. You only performs product search based on product ID , product recommendation , product offering ,  prommotional product provision , product status enquiry , adding product notes in system. Do not do perform any other action.Return the result with ONLY a list of products and their prices, Append the word 'Successfully completed' at the start of the message. If you do not find any answer  for a query , respond with only `nothing found`. Do not mention to try again"
)

def product_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = product_agent.invoke(state)
    dummy_response = result["messages"][-1].content
    return Command(
        update={
            "messages": [
                HumanMessage(content=dummy_response, name="Complaint")
            ]
        },
        goto="supervisor",
    )

builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("Order", order_node)
builder.add_node("Complaint", complaint_node)
builder.add_node("Product", product_node)
graph = builder.compile()

# COMMAND ----------

# from IPython.display import display, Image

# display(Image(graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

from langchain_core.runnables import RunnableLambda
from mlflow.langchain.output_parsers import ChatCompletionsOutputParser


# parse the output from the graph to get the final message, and then format into ChatCompletions
def get_final_message(resp):
    return resp["messages"][-1]
  
graph_with_parser = graph | RunnableLambda(get_final_message) | ChatCompletionsOutputParser()

# COMMAND ----------

import mlflow 
mlflow.models.set_model(graph_with_parser)

# COMMAND ----------

# graph_with_parser.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What is the status of my Order ORD342?"
#             }
#         ]
#     }
# )

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Examples

# COMMAND ----------

# graph_with_parser.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Recommend me some headphone products for my nieces upcoming birthday"
#             }
#         ]
#     }
# )

# COMMAND ----------

# user_query = "Recommend me some headphone products for my nieces upcoming birthday"
# for s in graph.stream(
#     {"messages": [("user", user_query)]}, subgraphs=True
# ):
#     print(s)
#     print("----")

# COMMAND ----------

# user_query = "What is the status of my complaint for my Order ORD342?"
# for s in graph.stream(
#     {"messages": [("user", user_query)]}, subgraphs=True
# ):
#     print(s)
#     print("----")

# COMMAND ----------


# user_query = "Provide some details of the product that i order for Order ORD342?"
# for s in graph.stream(
#     {"messages": [("user", user_query)]}, subgraphs=True
# ):
#     print(s)
#     print("----")

# COMMAND ----------


