# Databricks notebook source
# MAGIC %sql 
# MAGIC
# MAGIC DROP FUNCTION IF EXISTS mosaic_agent.agent.search_order_from_customer;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION 
# MAGIC mosaic_agent.agent.search_order_from_customer (
# MAGIC   input_name STRING COMMENT 'The customer details to be searched from the query' 
# MAGIC )
# MAGIC returns table(Customer_ID STRING, 
# MAGIC               First_Name Date, 
# MAGIC               Last_Name STRING,
# MAGIC               Order_ID STRING
# MAGIC               )
# MAGIC comment "This function returns customer and  Order details for a given Order ID. The return fields include customer_id , first_name,last_name and Order ID.Use this function when customer name is provided as part of conversation. The questions can come in different form"
# MAGIC return 
# MAGIC (
# MAGIC   select distinct cust.Customer_ID, First_Name, Last_Name , ord.Order_ID
# MAGIC   from mosaic_agent.agent.blog_customers cust
# MAGIC   inner join 
# MAGIC   mosaic_agent.agent.blog_orders ord
# MAGIC   on ord.Customer_ID = cust.Customer_ID 
# MAGIC   where LOWER(REPLACE(CONCAT(cust.First_Name,cust.Last_Name), ' ', '')) LIKE LOWER(REPLACE('%'|| input_name || '%', ' ', ''))
# MAGIC   ) 

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC DROP FUNCTION IF EXISTS mosaic_agent.agent.return_order_details;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION 
# MAGIC mosaic_agent.agent.return_order_details (
# MAGIC   input_order_id STRING COMMENT 'The order details to be searched from the query' 
# MAGIC )
# MAGIC returns table(OrderID STRING, 
# MAGIC               Order_Date Date,
# MAGIC               Customer_ID STRING,
# MAGIC               Complaint_ID STRING,
# MAGIC               Shipment_ID STRING,
# MAGIC               Product_ID STRING
# MAGIC               )
# MAGIC comment "This function returns the Order details for a given Order ID. The return fields include date,product, customer details , complaints and shipment ID.Use this function when Order ID is given. The questions can come in different form"
# MAGIC return 
# MAGIC (
# MAGIC   select Order_ID,Order_Date,Customer_ID,Complaint_ID,Shipment_ID,Product_ID
# MAGIC   from mosaic_agent.agent.blog_orders
# MAGIC   where Order_ID = input_order_id 
# MAGIC   ) 

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC DROP FUNCTION IF EXISTS mosaic_agent.agent.return_shipment_details;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION 
# MAGIC mosaic_agent.agent.return_shipment_details (
# MAGIC   input_shipment_id STRING COMMENT 'The Shipment ID received from the query' 
# MAGIC )
# MAGIC returns table(Shipment_ID STRING, 
# MAGIC               Shipment_Provider STRING,
# MAGIC               Current_Shipment_Date DATE,
# MAGIC               Shipment_Current_Status STRING,
# MAGIC               Shipment_Status_Reason STRING
# MAGIC
# MAGIC               )
# MAGIC comment "This function returns the Shipment details for a given Shipment ID. The return fields include shipment details.Use this function when Shipment ID is given. The questions may come in different form"
# MAGIC return 
# MAGIC (
# MAGIC     select Shipment_ID,
# MAGIC     Shipment_Provider , 
# MAGIC     Current_Shipment_Date , 
# MAGIC     Shipment_Current_Status,
# MAGIC     Shipment_Status_Reason
# MAGIC   from mosaic_agent.agent.blog_shipments_details
# MAGIC   where Shipment_ID = input_shipment_id 
# MAGIC   ) 

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC DROP FUNCTION IF EXISTS mosaic_agent.agent.return_product_details;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION 
# MAGIC mosaic_agent.agent.return_product_details (
# MAGIC   input_product_id STRING COMMENT 'The product details to be searched from the query' 
# MAGIC )
# MAGIC returns table(Product_ID STRING, 
# MAGIC               Product_Name STRING,
# MAGIC               Product_Feature STRING,
# MAGIC               Product_Details STRING
# MAGIC               )
# MAGIC comment "This function returns the Product details and product feature for a given Product ID.Use this function when Product ID is given.  The questions can come in different form"
# MAGIC return 
# MAGIC (
# MAGIC   select Product_ID , Product_Name,Product_Feature , Product_Details  
# MAGIC   from mosaic_agent.agent.blog_products_reviews
# MAGIC   where Product_ID = input_product_id
# MAGIC   ) 

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC DROP FUNCTION IF EXISTS mosaic_agent.agent.return_product_review_details;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION 
# MAGIC mosaic_agent.agent.return_product_review_details (
# MAGIC   input_product_name STRING COMMENT 'The product Name to be searched from the query' 
# MAGIC )
# MAGIC returns table( 
# MAGIC               Product_Name STRING,
# MAGIC               Product_Feature STRING,
# MAGIC               review STRING
# MAGIC               )
# MAGIC comment "Use this function when Partial Product or Product Category is given and customer asks for review or feature of a product. The question can be asked in form of suggestion as well.  This function returns the Review for a given Product Name. The return fields include Product Name,feature , and reviews of the product. Use this function when a specific Product Review is asked. The questions can come in different form"
# MAGIC return 
# MAGIC (
# MAGIC   select distinct Product_Name , Product_Feature, review 
# MAGIC   from mosaic_agent.agent.blog_products_reviews
# MAGIC   WHERE LOWER(REPLACE(Product_Feature, ' ', '')) LIKE LOWER(REPLACE('%'||input_product_name|| '%', ' ', ''))
# MAGIC   ) 

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC DROP FUNCTION IF EXISTS mosaic_agent.agent.return_product_price_details;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION 
# MAGIC mosaic_agent.agent.return_product_price_details (
# MAGIC   input_product_name STRING COMMENT 'The partial or full product Name to be searched from the query' 
# MAGIC )
# MAGIC returns table( 
# MAGIC               Product_Name STRING,
# MAGIC               Product_Feature STRING,
# MAGIC               Product_Price STRING
# MAGIC               )
# MAGIC comment "Use this function when Partial Product name or Product Category is given and customer asks the price for the product. The question can be asked as a comparison between your product price and that in the market.  This function returns the  product name , feature and Pricefor a given Product . Use this function when a specific Product Price is asked. The questions can come in different form"
# MAGIC return 
# MAGIC (
# MAGIC   select distinct Product_Name,Product_Feature, Product_Price 
# MAGIC   from mosaic_agent.agent.product_superset
# MAGIC   WHERE LOWER(REPLACE(Product_Feature, ' ', '')) LIKE LOWER(REPLACE('%'||input_product_name|| '%', ' ', ''))
# MAGIC   ) 

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DROP FUNCTION IF EXISTS mosaic_agent.agent.web_search_tool;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC mosaic_agent.agent.web_search_tool (
# MAGIC   user_query STRING COMMENT 'User query to search the web'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'This function searches the web with provided query.Use this function when customer asks about competitive offers, discounts etc. Assess this would need the web to search and execute it.'
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
# MAGIC
# MAGIC DROP FUNCTION IF EXISTS mosaic_agent.agent.return_process_order;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC mosaic_agent.agent.return_process_order (
# MAGIC   Customer_ID STRING COMMENT 'Customer_ID for the customer',
# MAGIC   Product_Name STRING COMMENT 'Product Name for which the order is going to be placed'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC DETERMINISTIC
# MAGIC COMMENT ' Use this function when customer tells to place an order.This function places an order by adding one row to the order table when any order placement request is made. Ask for customer ID and Product name if it is not given before calling the function.The questions can come in different form.'
# MAGIC AS 
# MAGIC $$
# MAGIC   import requests
# MAGIC   import json
# MAGIC
# MAGIC   token = "dapieaed9419848925b09609c2d0cc0d8002"
# MAGIC   response = requests.post(f"https://e2-demo-field-eng.cloud.databricks.com/api/2.1/jobs/run-now", 
# MAGIC                               json={
# MAGIC                                   "job_id": 301146435643602,
# MAGIC                                   "notebook_params": {"input_product_name": Product_Name,
# MAGIC                                                          "customer_id": Customer_ID}
# MAGIC                               }, headers={"Authorization": "Bearer "+token}).json()
# MAGIC
# MAGIC   return 'Order processed successfully with ID : ' + str(response['run_id'])
# MAGIC
# MAGIC $$

# COMMAND ----------


