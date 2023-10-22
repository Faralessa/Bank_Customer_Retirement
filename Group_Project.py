# pip install streamlit pandas matplotlib seaborn plotly sqlalchemy

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sqlalchemy import create_engine
import openai
import sqlite3
import csv

# Convert CSV to SQLite
def csv_to_sqlite(csv_file, db_name, table_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Read CSV and insert into SQLite table
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        header = [f'"{col.replace(" ", "_").replace("/", "_")}"' if col[0].isdigit() else col.replace(" ", "_").replace("/", "_") for col in header]  # Modify column names
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{col} TEXT' for col in header])})")
        cursor.executemany(f"INSERT INTO {table_name} VALUES ({', '.join(['?'] * len(header))})", csv_reader)

    conn.commit()
    conn.close()

# Convert CSV to SQLite
csv_to_sqlite("Bank_Customer_retirement.csv", "customer_data.db", "customer_table")

# Load data from SQLite
conn = sqlite3.connect('customer_data.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM customer_table")
df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
conn.close()

# Title
st.title('Bank Customer Retirement Data Analysis')

# Introduction
intro = """
As a collaborative data science team at a leading New York City bank, we aim to build a predictive model to determine a customer's retirement readiness. Our analysis involves exploring essential customer data, including age and 401K savings, to better understand the key factors influencing retirement preparedness. Through statistical analysis, correlation assessments, SQL querying, and GPT-3.5 integration, we seek to derive actionable insights that will inform the development of an accurate predictive model.
"""
st.write(intro)

# Show dataset
st.subheader('Dataset')
st.write(df)

# Drop the 'Customer ID' column
df.drop('Customer ID', axis=1, inplace=True)

# Round the numeric columns to 2 digits after the decimal point
df = df.round(2)

# Add tabs
tabs = ["Data Analysis", "Correlation Analysis", "SQL Query", "GPT-3.5 Integration"]
choice = st.sidebar.radio("Select a tab", tabs)

# Data Analysis tab
if choice == "Data Analysis":
    st.subheader('Data Analysis')

    # Mention the reason for dropping the 'Customer ID' column
    st.write("We have dropped the 'Customer ID' column before the analysis to eliminate any potential bias or misinterpretation that could arise from the analysis of an identifier column.")

    # Set display format for the DataFrame
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # Statistical summary
    st.subheader('Statistical Summary')
    st.write(df.describe().round(2))

    # Explanation of the statistical summary
    st.subheader('Explanation of the Statistical Summary')
    st.write("""
    The dataset analysis reveals essential insights for predicting a customer's retirement status. Firstly, the average age of the customers is approximately 46.76 years, with a range from 25 to 70 years. Understanding the relationship between age and retirement readiness is pivotal, as it provides valuable insights into how different age groups may approach retirement planning. Additionally, the significant variability in 401K savings, indicated by the standard deviation of $187,675.82, highlights the diverse financial preparedness among customers. This suggests that individuals with higher 401K savings may be better positioned for retirement, underlining the critical role of financial stability in retirement planning. 

    Moreover, the balanced distribution of retired and non-retired individuals within the dataset, as indicated by an equal representation of 0s and 1s in the 'Retire' feature, underscores the necessity of considering both groups when developing the predictive model. Ensuring that the model captures insights from both retired and non-retired individuals accurately is imperative for creating a comprehensive framework for predicting retirement readiness. By leveraging these key insights, a robust predictive model can be developed, taking into account the crucial factors of age, financial stability, and the overall distribution of retired and non-retired individuals in the dataset.
    """)

    # Plotly Histogram
    st.subheader('Age Distribution Histogram')

    import plotly.express as px

    fig = px.histogram(df, x='Age', nbins=15)
    st.plotly_chart(fig)

        # Explanation of Age Distribution
    st.write("The concentration of approximately 113 customers aged between 42.5 and 47.49, as depicted by the Plotly histogram, holds significance for our data science team. Understanding the distribution of customers within this key age bracket is crucial in developing an accurate predictive model for assessing retirement readiness. By focusing on this prominent customer segment, we can tailor our predictive model to provide targeted insights and personalized financial strategies, ensuring effective retirement planning for individuals within this core age group.")

# Correlation Analysis tab
elif choice == "Correlation Analysis":
    st.subheader('Correlation Analysis')
    corr = df.corr().round(2)
    st.write(corr)

    # Display correlation as an interactive heatmap
    st.subheader('Correlation Heatmap')
    import plotly.figure_factory as ff
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='Viridis',
        showscale=True,
    )
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig)

    # Include insights under the heatmap
    st.write("""
    The correlation matrix underscores essential relationships within the dataset. Firstly, the moderate positive correlation of 0.58 between 'Age' and '401K Savings' suggests that as a customer's age increases, their 401K savings tend to increase as well. This connection implies that older individuals are more likely to have higher retirement savings, emphasizing the importance of age in financial planning for retirement.

    Secondly, the strong positive correlation of 0.73 between 'Age' and 'Retire' suggests that as customers grow older, the probability of being retired increases. This correlation emphasizes the expected association between advancing age and the likelihood of retirement, reflecting the natural trend of transitioning into retirement as individuals age.

    Moreover, the substantial positive association of 0.78 between '401K Savings' and 'Retire' highlights the crucial role of financial preparedness in retirement. The correlation suggests that higher 401K savings are strongly linked to a higher likelihood of being retired, underlining the significance of adequate financial planning and stable retirement savings in ensuring a comfortable retirement.
    """)

    # Age distribution among customers who retired or not
    st.subheader('Age Distribution by Retirement Status')
    retired = df[df['Retire'] == 1]['Age']
    not_retired = df[df['Retire'] == 0]['Age']

    fig, ax = plt.subplots()
    sns.histplot(retired, color='skyblue', label='Retired', kde=False, bins=30)
    sns.histplot(not_retired, color='salmon', label='Not Retired', kde=False, bins=30)
    plt.legend()
    st.pyplot(fig)

    # 401K Savings distribution among customers who retired or not
    st.subheader('401K Savings Distribution by Retirement Status')
    retired_savings = df[df['Retire'] == 1]['401K Savings']
    not_retired_savings = df[df['Retire'] == 0]['401K Savings']

    fig, ax = plt.subplots()
    sns.histplot(retired_savings, color='skyblue', label='Retired', kde=False, bins=30)
    sns.histplot(not_retired_savings, color='salmon', label='Not Retired', kde=False, bins=30)
    plt.legend()
    st.pyplot(fig)

    # Seaborn
    st.subheader('Seaborn Visualization')
    fig, ax = plt.subplots()
    sns.boxplot(x='Retire', y='401K Savings', data=df, ax=ax)
    st.pyplot(fig)

    # Plotly
    st.subheader('Plotly Visualization')
    fig = px.scatter(df, x='Age', y='401K Savings', color='Retire')
    st.plotly_chart(fig)

# SQL Query tab
elif choice == "SQL Query":
    st.subheader('SQL Integration')

    # Save DataFrame to SQLite database
    conn = sqlite3.connect('customer_data.db')
    df.to_sql('customer_table', conn, if_exists='replace', index=False)

    # Example query
    example_query = '''SELECT Age, "401K Savings"
    FROM customer_table
    WHERE "401K Savings" > 800000
    ORDER BY Age
    LIMIT 5;'''
    st.markdown(f"_Example:_ {example_query}")

    # Input for SQL query
    user_query = st.text_area("Enter your SQL query:", value="", height=100)

    # Execute the query and display results
    if st.button('Run Query'):
        try:
            result = conn.execute(user_query).fetchall()
            column_names = [description[0] for description in conn.execute(user_query).description]
            st.write("Results of the SQL Query:")
            st.table(pd.DataFrame(result, columns=column_names))
        except Exception as e:
            st.write(f"An error occurred: {e}")

    # Close the connection
    conn.close()

# GPT-3.5 Integration tab
elif choice == "GPT-3.5 Integration":
    st.subheader('GPT-3.5 Integration')

    # Load OpenAI key from file
    with open('key.txt', 'r') as file:
        openai_key = file.read().strip()

    # Set OpenAI API key for GPT-3.5
    openai.api_key = openai_key

    # Introduction to GPT-3.5 integration
    st.write("Considering the crucial interplay of age, financial readiness, and retirement planning, we invite you to delve deeper into the dynamics of securing a stable financial future. With our ongoing analysis highlighting key trends in retirement preparation, we are eager to address any inquiries or uncertainties you may have about optimizing your financial plans for a comfortable retirement. Feel free to ask any questions or seek personalized insights to ensure a seamless transition into this significant phase of your life.")

    # Example related question in smaller font
    st.markdown("*Example:* What are some effective strategies for maximizing my 401K savings as I approach retirement age?", unsafe_allow_html=True)

    # Get user input
    user_input = st.text_input("Enter your question:", value='')

    # Call GPT-3.5 API
    if user_input:
        response = openai.Completion.create(
            engine="davinci",
            prompt=user_input,
            max_tokens=150
        )
        st.write("*GPT-3.5's Response:*")
        st.write(response.choices[0].text)
