# Databricks notebook source
# MAGIC %pip install --upgrade mlflow tensorflow langchain-community
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### Generate a MLFlow Function 

# COMMAND ----------

import mlflow.pyfunc
import pandas as pd
import os 
from langchain_community.utilities import GoogleSerperAPIWrapper

mlflow.set_registry_uri("databricks-uc")

CATALOG_NAME = "mosaic_agent"
SCHEMA_NAME = "agent"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.web_search"


class SerperAPIModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        os.environ["SERPER_API_KEY"] = "a6b875b80be0db48efb153468eadab3600252fa1"
        self.search_tool = GoogleSerperAPIWrapper()

    def predict(self, context, model_input):
        search_results = self.search_tool.run(model_input.iloc[0])
        return search_results

# Log the model using MLflow and register it into Unity Catalog
if __name__ == "__main__":
    with mlflow.start_run(run_name='serper_api_model') as run:
        mlflow.pyfunc.log_model(
            artifact_path="serper_api_model",
            python_model=SerperAPIModel(),
            registered_model_name=MODEL_NAME,
            input_example=pd.DataFrame([["Whats the weather in SF"]])
        )

# COMMAND ----------

# DBTITLE 1,Test the Function through MLFlow
from mlflow.models import validate_serving_input

model_uri = 'runs:/6696b96f2f9a4b19ab3928a6e00b0292/serper_api_model'

# The model is logged with an input example. MLflow converts
# it into the serving payload format for the deployed model endpoint,
# and saves it to 'serving_input_payload.json'

serving_payload = """{
  "dataframe_split": {
    "data": [
      [
        "What is the latest Deals going on Australia for Headphones ? "
      ]
    ]
  }
}"""

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Create Tool 

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC mosaic_agent.agent.web_search_tool (
# MAGIC   user_query STRING COMMENT 'User query to search the web'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'This function searches the web with provided query'
# MAGIC AS 
# MAGIC $$
# MAGIC
# MAGIC   import requests
# MAGIC   import json
# MAGIC   import numpy as np
# MAGIC   import pandas as pd
# MAGIC   import json
# MAGIC
# MAGIC   url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/web_search_tool_API/invocations'
# MAGIC   headers = {'Authorization': f'Bearer dapia5b5af084fc6a35210a672fe71e229f4', 'Content-Type': 'application/json'}
# MAGIC
# MAGIC   response = requests.request(method='POST', headers=headers, url=url, data=json.dumps({
# MAGIC                               "dataframe_split": {"data": [[user_query]]}}))
# MAGIC
# MAGIC   return response.json()['predictions']
# MAGIC
# MAGIC $$

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT daiwt_cafe_retail.ai.search_tool("What are the latest offers running on Headpones in Sydney ? ") as response

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT daiwt_cafe_retail.ai.search_tool("What are the product warranty for Samsung Galaxy S20 Ultra?  ") as response

# COMMAND ----------


