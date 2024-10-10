# Databricks notebook source
# MAGIC %pip install  --upgrade --quiet databricks-sdk langchain-community mlflow langchain-openAI Faker transformers
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Dataset Generation

# COMMAND ----------

import random
import string
import pandas as pd
from faker import Faker
from transformers import pipeline
from databricks.sdk import WorkspaceClient
from langchain_openai import ChatOpenAI
import json 
import re 

# Initialize faker and random seed
fake = Faker()
random.seed(42)

# Define some product names and categories
product_catalog = [
    {"Product Name": "Wireless Headphones", "Category": "Electronics"},
    {"Product Name": "Bluetooth Speaker", "Category": "Electronics"},
    {"Product Name": "Smartphone", "Category": "Electronics"},
    {"Product Name": "Espresso Machine", "Category": "Kitchen Appliances"},
    {"Product Name": "Rice Cooker", "Category": "Kitchen Appliances"},
    {"Product Name": "LED TV", "Category": "Home Entertainment"},
    {"Product Name": "Gaming Laptop", "Category": "Computers"},
    {"Product Name": "Personal Computer", "Category": "Computers"},
    {"Product Name": "Blender", "Category": "Kitchen Appliances"},
    {"Product Name": "Electric Kettle", "Category": "Kitchen Appliances"},
    {"Product Name": "Electric Bike", "Category": "Outdoor Equipment"},
    {"Product Name": "Camera", "Category": "Outdoor Equipment"}
]

# COMMAND ----------

# DBTITLE 1,OSS LLM
# Initialize a Llama text generator
llm = ChatOpenAI(
  base_url=f"{WorkspaceClient().config.host}/serving-endpoints/",
  api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
  model="databricks-meta-llama-3-1-70b-instruct"
)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Product

# COMMAND ----------

def generate_product_description(llm,product_name, product_category):

    prompt = f"""Always generate a json object with two key value pairs for product {product_name} in the category {product_category}. The response should ONLY contain the valid JSON object without any other content. 
            The output should strictly be in json format with no additional characters.

            1. "product_feature": A concise description (within 20 words) of the product in specific category, specify the brand and key feature.
            2. "feature_details": A description (within 100 words) of the product’s features, including availability in Australia. Do not add any special characters. Include adjectives in the feature details column if needed.Do not add any special characters. 
          
          Be specific on features and brand in the product description ,do not use any adjectives. 
          
          """
    
    # Use llama-3.1 to generate product description
    result = llm.invoke(prompt,temperature = 0.01)
    
    # Extract the generated text and return
    return result

def generate_product_review(llm,product_name, product_feature):

    prompt = f"""generate fake product reivew for product in JSON format {product_name} with detail as //{product_feature}// . The response  should exactly be the below json format and nothing else. The output should strictly be in json format with no additional characters. The three attributes key-value pairs in the json should  be: 
           1. product_name: Name of the product
           2. product_review: review string within 200 words
           3. review date: Random Date between 2024-01-01 and 2024-06-01.
          
      Be specific on features and brand in the product description ,do not use any adjectives.
          
          """

    response = llm.invoke(prompt, temperature=0.01)
    cleaned_string = json.loads(re.sub(r'```|\n', '', response.content))

    return cleaned_string

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Incident

# COMMAND ----------

def generate_complaint_id():
    # 30% chance to have a complaint ID
    if random.random() <= 0.3:
        
        return f'inc{random.randint(0, 9999):04d}'
    else:
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Order

# COMMAND ----------

def generate_order_dataset(num_orders=100):
    data = []
    unique_products = set()
    unique_customers = set()
    
    for _ in range(num_orders):
        order_id = 'ORD' + str(random.randint(1,1000))
        order_date = fake.date_between(start_date='-1y', end_date='today')
        status = random.choice(['Complete','Placed','Pending', 'Shipped', 'Delivered', 'Cancelled'])

        # Select a random product from the catalog
        product = random.choice(product_catalog)
        product_id = 'PRD' + str(random.randint(1,1000))
        unique_products.add(product_id)

        # Generate a meaningful product description using GPT-2
        prod_response = generate_product_description(llm,product['Product Name'], product['Category'])

        # Remove backticks and newlines
        cleaned_string = re.sub(r'```|\n', '', prod_response.content)

        feature = json.loads(cleaned_string)['product_feature']
        details = json.loads(cleaned_string)['feature_details']
        
        order_quantity = random.randint(1, 10)
        order_price = round(random.uniform(10.5, 500.0), 2)
        customer_id = 'CUST' + str(random.randint(1,1000))
        unique_customers.add(customer_id)

        shipment_id = 'SHP' + str(random.randint(1,1000))
        
        # Generate a Complaint ID with a 30% chance
        complaint_id = generate_complaint_id()

        data.append({
            'Order_ID': order_id,
            'Order_Date': order_date,
            'Status': status,
            'Product_ID': product_id,
            'Product_Category':product['Category'],
            'Product_Name': product['Product Name'],
            'Product_Feature': feature, 
            'Product_Details': details,
            'Order_Quantity': order_quantity,
            'Order_Price': order_price,
            'Customer_ID': customer_id,
            'Shipment_ID': shipment_id,
            'Complaint_ID': complaint_id
        })
    
    df = pd.DataFrame(data)
    return df, list(unique_products), list(unique_customers)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Order + Product

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, format_string

if __name__ == '__main__':

  # Generate order dataset
  order_dataset, unique_products, unique_customers = generate_order_dataset(100)
  spark.createDataFrame(order_dataset).write.mode("overwrite").saveAsTable("mosaic_agent.agent.blog_orders")

  product_feature = ["Product_ID","Product_Name", "Product_Feature", "Product_Details"]
  product_cols = ["Product_ID","Product_Category_ID","Product_Name", "Product_Feature", "Product_Details","review","review_date"]

  #Read the order table
  order_df = spark.table("mosaic_agent.agent.blog_orders")

  #create product table with relevant product details
  product_df = order_df.select(product_feature) \
                    .withColumn("Product_Category_ID", format_string("PRD%04d", 
                                                            monotonically_increasing_id() % 1000000 + 100)) \
                    .toPandas() \
                    .drop_duplicates(product_feature) \
                  
  
  #Generate Product Reviews for each product
  review_list = []
  review_date_list = [] 

  for i in range(0 ,len(product_df)):

    #Generate product reviews for each product
    review_response = generate_product_review(llm,
                                              product_df.iloc[i]['Product_Name'],
                                              product_df.iloc[i]['Product_Feature'])

    #Append synthetic product review to the list
    try:
      review_list.append(review_response['product_review']) 
      review_date_list.append(review_response['review_date'])
    except:
      review_list.append(review_response[0]['product_review']) 
      review_date_list.append(review_response[0]['review_date'])

  product_df['review'] = review_list
  product_df['review_date'] = review_date_list     

  spark.createDataFrame(product_df) \
        .select(product_cols).write.mode('overwrite') \
        .saveAsTable('mosaic_agent.agent.products_reviews')

  display(product_df)   

# COMMAND ----------

# spark.sql("drop table mosaic_agent.agent.blog_products_reviews")

product_cols = ["Product_ID","Product_Category_ID","Product_Name", "Product_Feature", "Product_Details","review","review_date"]

spark.createDataFrame(product_df)\
  .select(product_cols).write.mode('overwrite') \
    .saveAsTable('mosaic_agent.agent.blog_products_reviews') 

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Incident

# COMMAND ----------


import pandas as pd
from random import choice, randint
from datetime import datetime, timedelta
import numpy as np 

# Define the complaint data
complaint_data = {
    'Complaint_ID': ['inc0106', 'inc0964', 'inc1307', 'inc1796', 'inc3681', 
                     'inc4092', 'inc4889', 'inc5820', 'inc7962', 'inc8935'],
    'Product': ['Camera', 'Wireless Headphones', 'Electric Kettle', 'LED TV', 
                'Gaming Laptop', 'Espresso Machine', 'Camera', 
                'Bluetooth Speaker', 'Espresso Machine', 'Wireless Headphones']
}

# Random status options
complaint_status = ['Open', 'Closed', 'In Progress', 'Resolved', 'ON HOLD']

# Generate random dates within the past year for Complaint_Date
def generate_random_date():
    start_date = datetime.now() - timedelta(days=365)
    random_days = randint(0, 365)
    return start_date + timedelta(days=random_days)

def generate_complaint_note(product_name, product_details):

    prompt = f"""Generate one single complaint note for product {product_name} with details  {product_details} on different dates.Must Only keep the complaint as output.Do not include sentence like "here is the complaint...." , keep only the complaint as output.Be specific on customer frustration and delay. remove all special characters from the output.It should not include any PII or date information """
    
    # Use GPT-2 to generate product description
    result = llm.invoke(prompt, temperature=0.1)
    
    # Extract the generated text and return
    return json.loads(result.json())['content']

if __name__ == '__main__':

    # Create the complaint dataset
    order = spark.table("mosaic_agent.agent.blog_orders").toPandas()
    complaint_notes = order[~order['Complaint_ID'].isnull()][['Complaint_ID', 'Order_ID','Product_Name','Product_Details', 'Product_Feature']]

    # Define the function to generate the complaint note
    for i in complaint_notes.index:
        complaint_notes.loc[i, 'Complaint_Note'] = generate_complaint_note(
            complaint_notes.loc[i, 'Product_Name'], 
            complaint_notes.loc[i, 'Product_Details']
            )

    # Generate random dates between 2024-01-01 and 2024-06-01
    complaint_notes['Complaint_Date'] = pd.to_datetime(np.random.choice(( pd.date_range('2024-01-01', '2024-06-01')), len(complaint_notes)))

    complaint_notes['Complaint_Date'] = complaint_notes['Complaint_Date'].dt.date

    complaint_notes['Complaint_Status'] = np.random.choice(complaint_status, len(complaint_notes))

    # Create the final dataset and save to table
    spark.createDataFrame(complaint_notes) \
                .write.mode("overwrite").saveAsTable("mosaic_agent.agent.blog_complaints")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Customer 

# COMMAND ----------

import random
import pandas as pd
from faker import Faker

# Initialize Faker object
fake = Faker()

customer_list = [row['Customer_ID'] for row in spark.sql("select distinct Customer_ID from mosaic_agent.agent.blog_orders").collect()]

# Predefined customer IDs
customer_ids = customer_list

# Function to generate random customer data
def generate_customer_data(num_customers=len(customer_ids)):
    data = []
    for _ in range(num_customers):
        customer_id = random.choice(customer_ids)
        first_name = fake.first_name()
        last_name = fake.last_name()
        address = fake.address().replace("\n", ", ")
        email = fake.email()
        phone_number = f'{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}'
        data.append({
            'Customer_ID': customer_id,
            'First_Name': first_name,
            'Last_Name': last_name,
            'Address': address,
            'Email': email,
            'Phone_Number': phone_number
        })
    return pd.DataFrame(data)

# Generate customer table
customer_table = generate_customer_data()

#Store the dataframe in Delta
spark.createDataFrame(customer_table).write.mode("overwrite").saveAsTable("mosaic_agent.agent.blog_customers")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Shipment 

# COMMAND ----------

import pandas as pd
import random
from datetime import datetime, timedelta

# List of Shipment IDs
shipment_ids = [
'SHP850','SHP864','SHP774','SHP670','SHP850','SHP816','SHP886','SHP822','SHP198','SHP983',
'SHP256','SHP610','SHP562','SHP212','SHP226','SHP818','SHP607','SHP852','SHP643','SHP851','SHP389',
]

# Shipment Providers
providers = ["DHL", "FedEx", "UPS", "Australia Post", "TNT"]

# Shipment Statuses
statuses = ["In Transit", "Delivered", "Pending", "Out for Delivery", "In Warehouse","Arrived at Airport"]

# Generate shipment details
shipments_data = {
    "Shipment_ID": shipment_ids,
    "Shipment_Provider": [random.choice(providers) for _ in shipment_ids],
    "Current_Shipment_Date": [(datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d') for _ in shipment_ids],
    "Shipment_Current_Status": [random.choice(statuses) for _ in shipment_ids]
}

# Create DataFrame
shipments_df = pd.DataFrame(shipments_data)

spark.createDataFrame(shipments_df).write.mode("overwrite").saveAsTable("mosaic_agent.agent.blog_shipments")

display(shipments_df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC Create or replace table mosaic_agent.agent.blog_shipments_details as ( 
# MAGIC SELECT Shipment_ID,
# MAGIC             Shipment_Provider , Current_Shipment_Date , Shipment_Current_Status,
# MAGIC             ai_query('databricks-meta-llama-3-1-70b-instruct',
# MAGIC                     'generate a fake shipment status reason in one sentence based on shipment status. ' || Shipment_Current_Status ||
# MAGIC                     'Do not include the word fake in it. Just return a string on Shipment Status Reason.'
# MAGIC                     ) AS Shipment_Status_Reason
# MAGIC FROM mosaic_agent.agent.blog_shipments
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC with sample_set as (
# MAGIC   select * from mosaic_agent.agent.blog_orders
# MAGIC   LIMIT 1
# MAGIC )
# MAGIC SELECT product_name,product_category,
# MAGIC   ai_query(
# MAGIC     'databricks-meta-llama-3-1-70b-instruct',
# MAGIC     'generate fake product review for product in JSON format ' || product_name || ' with ' || product_category || ' category. The response should exactly be the below json format and nothing else.'
# MAGIC     )
# MAGIC   FROM sample_set
