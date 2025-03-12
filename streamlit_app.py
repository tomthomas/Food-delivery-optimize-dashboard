import streamlit as st
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (replace with actual file path if needed)
df = pd.read_csv("data/food_order.csv")

# Create in-memory SQLite database
conn = sqlite3.connect(":memory:")
df.to_sql("orders", conn, index=False, if_exists="replace")

# Define SQL queries
sql_queries = {
    "Total Orders": "SELECT COUNT(*) AS total_orders FROM orders;",
    "Average Order Cost": "SELECT AVG(cost_of_the_order) AS average_order_cost FROM orders;",
    "Highest-Rated Restaurants": """
        SELECT restaurant_name, COUNT(*) AS max_rating_count, AVG(CAST(rating AS FLOAT)) AS avg_rating
        FROM orders
        WHERE rating != 'Not given' 
        AND CAST(rating AS FLOAT) = (SELECT MAX(CAST(rating AS FLOAT)) FROM orders WHERE rating != 'Not given')
        GROUP BY restaurant_name
        ORDER BY max_rating_count DESC
        LIMIT 10;
    """,
    "Revenue by Restaurant": """
        SELECT restaurant_name, SUM(cost_of_the_order) AS total_revenue
        FROM orders 
        GROUP BY restaurant_name 
        ORDER BY total_revenue DESC
        LIMIT 10;
    """,
    "Average Delivery Time per Cuisine": """
        SELECT cuisine_type, AVG(delivery_time) AS avg_delivery_time
        FROM orders 
        GROUP BY cuisine_type 
        ORDER BY avg_delivery_time DESC;
    """,
    "Average Delivery Time per Restaurant": """
        SELECT restaurant_name, AVG(delivery_time) AS avg_delivery_time
        FROM orders 
        GROUP BY restaurant_name 
        ORDER BY avg_delivery_time ASC
        LIMIT 20;
    """,
    "Peak Order days of the week": """
        SELECT day_of_the_week, COUNT(*) AS total_orders
        FROM orders
        GROUP BY day_of_the_week
        ORDER BY total_orders DESC;
    """,
    "Revenue by Cuisine type": """
        SELECT cuisine_type, SUM(cost_of_the_order) AS total_revenue
        FROM orders
        GROUP BY cuisine_type
        ORDER BY total_revenue DESC
        LIMIT 10;
    """,
    "Customer spending habits": """
        SELECT o.customer_id, 
            SUM(o.cost_of_the_order) AS total_spent,
            r.restaurant_name AS top_restaurant,
            r.cuisine_type AS top_cuisine
        FROM orders o
        JOIN (
                SELECT customer_id, restaurant_name, cuisine_type,
                ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY SUM(cost_of_the_order) DESC) AS rank
        FROM orders
        GROUP BY customer_id, restaurant_name, cuisine_type
    ) r
    ON o.customer_id = r.customer_id
    WHERE r.rank = 1
    GROUP BY o.customer_id
    ORDER BY total_spent DESC
    LIMIT 10;

    """,
    "Restuarants with long prep time but high ratings": """
        SELECT restaurant_name, 
       AVG(food_preparation_time) AS avg_prep_time, 
       AVG(CAST(rating AS FLOAT)) AS avg_rating
        FROM orders
        WHERE rating != 'Not given'
        GROUP BY restaurant_name
        HAVING avg_prep_time > 30 AND avg_rating >= 4.5
        ORDER BY avg_prep_time DESC;

    """
}

# ML Code for Predictions
# Preprocess data
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['cuisine_type'] = df['cuisine_type'].astype('category')
df['churn'] = df['rating'].apply(lambda x: 1 if x < 4 else 0)

# Group by customer for prediction features
df_customer = df.groupby('customer_id').agg({
    'cost_of_the_order': ['sum', 'mean'],
    'rating': 'mean',
    'cuisine_type': lambda x: x.mode()[0],  # Most frequent cuisine
    'food_preparation_time': 'mean',
    'delivery_time': 'mean',
    'churn': 'max'
}).reset_index()

df_customer.columns = ['customer_id', 'total_spent', 'avg_spent', 'avg_rating', 'favorite_cuisine', 'avg_food_prep_time', 'avg_delivery_time','churn']

# Train regression model for spending prediction
X_spend = df_customer[['avg_spent', 'avg_food_prep_time', 'avg_delivery_time']]
y_spend = df_customer['total_spent']
X_train, X_test, y_train, y_test = train_test_split(X_spend, y_spend, test_size=0.2, random_state=42)
spend_model = RandomForestRegressor()
spend_model.fit(X_train, y_train)

# Train classification model for churn prediction
X_churn = df_customer[['avg_rating', 'avg_food_prep_time', 'avg_delivery_time']]
y_churn = df_customer['churn']
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)
churn_model = RandomForestClassifier()
churn_model.fit(X_train_churn, y_train_churn)

# Train classification model for cuisine prediction
X_cuisine = df_customer[['avg_spent', 'avg_food_prep_time', 'avg_delivery_time']]
y_cuisine = df_customer['favorite_cuisine']
X_train_cuisine, X_test_cuisine, y_train_cuisine, y_test_cuisine = train_test_split(X_cuisine, y_cuisine, test_size=0.2, random_state=42)
cuisine_model = RandomForestClassifier()
cuisine_model.fit(X_train_cuisine, y_train_cuisine)

# Function for forecasting restaurant data (using ARIMA)
# Set minimum orders threshold
MIN_ORDERS = 30  


# Filter dataset to include only restaurants with at least MIN_ORDERS
restaurant_counts = df['restaurant_name'].value_counts()
valid_restaurants = restaurant_counts[restaurant_counts >= MIN_ORDERS].index
df_filtered = df[df['restaurant_name'].isin(valid_restaurants)]

def forecast_revenue_and_orders(restaurant, months):
    # Filter data for the selected restaurant
    restaurant_data = df_filtered[df_filtered['restaurant_name'] == restaurant].copy()
    
    if len(restaurant_data) < 10:  # Ensure enough data points
        return None, None
    
    # Generate order_date if missing
    if 'order_date' not in restaurant_data:
        restaurant_data['order_date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(
            np.random.randint(0, 365, size=len(restaurant_data)), unit='D'
        )

        restaurant_data = restaurant_data.sort_values(by='order_date')
        restaurant_data.set_index('order_date', inplace=True)

        # Time series: Orders and revenue
        orders_series = restaurant_data['order_id'].resample('M').count()
        revenue_series = restaurant_data['cost_of_the_order'].resample('M').sum()

        # ARIMA for Orders
        model_orders = ARIMA(orders_series, order=(5, 1, 0))
        model_orders_fit = model_orders.fit()
        forecast_orders = model_orders_fit.forecast(steps=months)

        # ARIMA for Revenue
        model_revenue = ARIMA(revenue_series, order=(5, 1, 0))
        model_revenue_fit = model_revenue.fit()
        forecast_revenue = model_revenue_fit.forecast(steps=months)

        return forecast_orders, forecast_revenue

# Streamlit UI
st.set_page_config(page_title="Food Delivery Analytics", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Run SQL Queries", "Run Machine Learning Insights"])

# Sidebar image placeholder
st.sidebar.image("data/food_delivery.png", caption="Food Delivery Analytics", use_container_width=True)

# Home Page
if page == "Home":
    st.title("🍽️ Food Delivery Analytics Dashboard")
    st.write(
        """
        Welcome to the **Food Delivery Analytics Dashboard**!  
        This app allows you to explore food order data using:
        - **SQL Queries** to extract insights.
        - **Machine Learning** to generate predictive insights.
        
        ### 📊 Dataset Overview:
        The dataset contains **customer food orders**, including:
        - **Order Cost** 💰
        - **Restaurant Name** 🏢
        - **Cuisine Type** 🍕🍣
        - **Food Preparation & Delivery Time** ⏳
        - **Customer Ratings** ⭐
        
        Use the sidebar to navigate!
        """
    )

# SQL Queries Page
elif page == "Run SQL Queries":
    st.title("🗄️ SQL Query Explorer")

    # Show dataset preview
    st.subheader("📋 Dataset Preview")
    st.write(df.head(10))  # Show first 10 rows

    # Dropdown for selecting SQL query
    query_option = st.selectbox("Choose a SQL Query:", list(sql_queries.keys()))

    if st.button("Run Query"):
        st.subheader("SQL Query:")
        st.code(sql_queries[query_option], language="sql")

        # Execute query
        result = pd.read_sql_query(sql_queries[query_option], conn)
        st.dataframe(result)  # Display query results

# ML Insights Page
elif page == "Run Machine Learning Insights":
    st.title("📊 Machine Learning Insights")

    # Predicting Customer Spending, Cuisine, and Churn
    st.header("💵 Customer Spending & Cuisine Prediction")

    customer_id = st.selectbox("Select Customer ID:", df_customer['customer_id'].unique())

    # Predict spending
    customer_data = df_customer[df_customer['customer_id'] == customer_id]
    spending_pred = spend_model.predict(customer_data[['avg_spent', 'avg_food_prep_time', 'avg_delivery_time']])[0]
   
    # Predict favorite cuisine
    cuisine_pred = cuisine_model.predict(customer_data[['avg_spent', 'avg_food_prep_time', 'avg_delivery_time']])[0]
   
    # Predict churn
    churn_pred = churn_model.predict(customer_data[['avg_rating', 'avg_food_prep_time', 'avg_delivery_time']])[0]
    churn_status = "Likely to churn" if churn_pred == 1 else "Likely to stay"
    
    # Create three columns for displaying insights
    col1, col2, col3 = st.columns(3)

    # Predicted Spending
    with col1:
        st.metric(label="Predicted Spending 💰", value=f"${spending_pred:.2f}")

    # Predicted Favorite Cuisine
    with col2:
        st.metric(label="Favorite Cuisine 🍽️", value=cuisine_pred)

    # Churn Prediction
    with col3:
        churn_status = "Likely to Churn ❌" if churn_pred == 1 else "Likely to Stay ✅"
        st.metric(label="Churn Prediction 🔄", value=churn_status)


    # Churn percentage
    churn_percentage = df['churn'].mean() * 100
    labels = ['Churned', 'Stayed']
    values = [churn_percentage, 100 - churn_percentage]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
    fig.update_layout(title="Overall Churn Percentage")
    st.plotly_chart(fig)

    # Streamlit UI
    st.header("🏢 Restaurant Order & Revenue Forecasting")

    # Dropdown with filtered restaurants
    restaurant_name = st.selectbox("Select Restaurant:", df_filtered['restaurant_name'].unique())
    forecast_months = st.number_input("Months to Forecast:", min_value=1, max_value=12, value=6)

    forecast_orders, forecast_revenue = forecast_revenue_and_orders(restaurant_name, forecast_months)

    if forecast_orders is not None:
        st.subheader(f"📉 Forecasted Orders for {restaurant_name}:")
        st.line_chart(forecast_orders)  # Use a line chart for better visualization

        st.subheader(f"💰 Forecasted Revenue for {restaurant_name}:")
        st.line_chart(forecast_revenue)
    else:
        st.warning("No data available for this restaurant. Try selecting another one.")
