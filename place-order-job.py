# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from datetime import datetime
import random
import string
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType, NullType
from pyspark.sql.functions import to_date
from datetime import datetime

# Assuming the types of the variables, adjust as necessary
order_schema = StructType([
        StructField("Order_ID", StringType(), True),
        StructField("Order_Date", DateType(), True),
        StructField("Status", StringType(), True),
        StructField("Product_ID", StringType(), True),
        StructField("Product_Category", StringType(), True),
        StructField("Product_Name", StringType(), True),
        StructField("Product_Feature", StringType(), True),
        StructField("Product_Details", StringType(), True),
        StructField("Order_Quantity", IntegerType(), True),
        StructField("Order_Price", FloatType(), True),
        StructField("Customer_ID", StringType(), True),
        StructField("Shipment_ID", StringType(), True),
        StructField("Complaint_ID", StringType(), True)  # or NullType() if it should be nullable
        ])

def extract_product_details (prd_name):
  product_table = spark.table('mosaic_agent.agent.blog_products_reviews')
  product_table.createOrReplaceTempView('product_table')

  prd_details = spark.sql("""select *
                              from product_table where 
            LOWER(REPLACE(Product_Feature, ' ', '')) 
            LIKE LOWER(REPLACE('%{0}%', ' ', ''))""".format(prd_name)).collect()[0] 

  return prd_details


# Function to generate a random Order ID
def generate_order_id():
    return 'ORD' + ''.join(random.choices(string.digits, k=3))

# Function to create a new row and insert into table_1
def insert_order(input_product_name, customer_id):

    #Extract all elements of products from product table 
    prd_details = extract_product_details(input_product_name) 
    # Generate values
    order_id = generate_order_id()
    temp_order_date = datetime.now().strftime('%Y-%m-%d')
    order_date = datetime.strptime(temp_order_date, "%Y-%m-%d").date()
    status = 'Pending'
    product_id = prd_details.Product_ID
    product_category = 'Kitchen Appliances'  # Example category
    product_name =  prd_details.Product_Name # Example product name
    product_feature = prd_details.Product_Feature     # Example feature
    product_details = prd_details.Product_Details  # Example details
    order_quantity = 1
    order_price = 79.99            
    shipment_id = None
    complaint_id = None
    
    # Create a DataFrame for the new row
    new_row = spark.createDataFrame([(order_id, order_date, status, product_id, product_category, 
                                      product_name, product_feature, product_details, 
                                      order_quantity, order_price, customer_id, shipment_id, complaint_id)],schema=order_schema)
    
    # Insert the row into the table (Assuming table_1 is already created as a temporary view or table in Spark)
    new_row.createOrReplaceTempView("new_row_temp")

    # Use Spark SQL to insert the row into the existing table
    spark.sql("""
        INSERT INTO mosaic_agent.agent.blog_orders
        SELECT * FROM new_row_temp
    """)

    print("Order inserted successfully.")

# COMMAND ----------

dbutils.widgets.text("input_product_name", "", "Input Product Name")
dbutils.widgets.text("customer_id", "", "Customer ID")

input_product_name = dbutils.widgets.get("input_product_name")
customer_id = dbutils.widgets.get("customer_id")

# Proceed with your function call
insert_order(input_product_name, customer_id)
