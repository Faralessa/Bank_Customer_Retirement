# pip install streamlit pandas matplotlib seaborn plotly sqlalchemy
# pip install openai

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sqlite3
import csv
import openai
import os

# Read CSV directly
df = pd.read_csv("Bank_Customer_retirement.csv")

# Title
st.title('Bank Customer Retirement Data Analysis')

# Introduction
intro = """
As a collaborative data science team at a leading New York City bank, we aim to build a predictive model to determine a customer's retirement readiness. Our analysis involves exploring essential customer data, including age and amount of savings, to better understand the key factors influencing retirement preparedness. Through statistical analysis, correlation assessments, SQL querying, and GPT-4 integration, we seek to derive actionable insights that will inform the development of an accurate predictive model.
"""
st.write(intro)

# Show original dataset
st.subheader('Original Dataset')
st.write(df)

# Convert DataFrame to SQLite
def df_to_sqlite(df, db_name, table_name):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

# Convert DataFrame to SQLite
df_to_sqlite(df, "customer_data.sqlite", "customer_table")

# Load data from SQLite
conn = sqlite3.connect('customer_data.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT * FROM customer_table")
df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
conn.close()

############################################################################################################
# Additional imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Split the data
X = df[['Age', '401K Savings']]
y = df['Retire']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)
############################################################################################################

# Add tabs
tabs = ["Data Analysis", "Correlation Analysis", "SQL Query", "GPT-4 Integration","Model Results", "Predictions"]
choice = st.sidebar.radio("Select a tab", tabs)

# Data Analysis tab
if choice == "Data Analysis":
    st.subheader('Data Analysis')

    # Drop the 'Customer ID' column
    df.drop('Customer ID', axis=1, inplace=True)

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
    The dataset analysis reveals essential insights for predicting a customer's retirement status. Firstly, the average age of the customers is approximately 46.76 years, with a range from 25 to 70 years. Understanding the relationship between age and retirement readiness is pivotal, as it provides valuable insights into how different age groups may approach retirement planning. Additionally, the significant variability in 401K savings, indicated by the standard deviation of $187,675.82, highlights the diverse financial preparedness among customers. This suggests that individuals with higher savings may be better positioned for retirement, underlining the critical role of financial stability in retirement planning. 

    Moreover, the balanced distribution of retired and non-retired individuals within the dataset, as indicated by an equal representation of 0s and 1s in the 'Retire' feature, underscores the necessity of considering both groups when developing the predictive model. Ensuring that the model captures insights from both retired and non-retired individuals accurately is imperative for creating a comprehensive framework for predicting retirement readiness. By leveraging these key insights, a robust predictive model can be developed, taking into account the crucial factors of age, financial stability, and the overall distribution of retired and non-retired individuals in the dataset.
    """)

    # Plotly Histogram
    st.subheader('Age Distribution Histogram')

    import plotly.express as px

    fig = px.histogram(df, x='Age', nbins=15)
    st.plotly_chart(fig)

    # Explanation of Age Distribution
    st.write("The concentration of approximately 114 customers aged between 42.5 and 47.49, as depicted by the Plotly histogram, holds significance for our data science team. Understanding the distribution of customers within this key age bracket is crucial in developing an accurate predictive model for assessing retirement readiness. By focusing on this prominent customer segment, we can tailor our predictive model to provide targeted insights and personalized financial strategies, ensuring effective retirement planning for individuals within this core age group.")

    # Age distribution among customers who retired or not
    st.subheader('Age Distribution by Retirement Status')
    retired = df[df['Retire'] == 1]['Age']
    not_retired = df[df['Retire'] == 0]['Age']

    fig, ax = plt.subplots()
    sns.histplot(retired, color='skyblue', label='Retired', kde=False, bins=30)
    sns.histplot(not_retired, color='salmon', label='Not Retired', kde=False, bins=30)
    plt.legend()
    st.pyplot(fig)
    
    # Short description under the figure
    st.write("This distribution illustrates a notable concentration of retired individuals in the late 40s and above, indicating a prevalent trend of retirement within this age group. The data underscores the significance of effective financial planning and retirement strategies, especially for customers approaching their late 40s and beyond, to ensure a secure and stable post-retirement financial situation.")

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

    # Short description under the figure
    st.write("The figures reveal that non-retired individuals tend to have savings below $600,000, with the highest concentration between $300,000 and $400,000. In contrast, retired individuals show a broader range of savings, starting from around $400,000 and extending up to $1,000,000, possibly due to effective long-term financial planning, increased investment opportunities, employer retirement benefits, and personal financial discipline. These factors collectively contribute to the growth of their savings over time.")

    # Plotly
    st.subheader('Plotly Visualization')
    fig = px.scatter(df, x='Age', y='401K Savings', color='Retire')
    st.plotly_chart(fig)

# Correlation Analysis tab
elif choice == "Correlation Analysis":
    st.subheader('Correlation Analysis')

    # Drop the 'Customer ID' column
    df.drop('Customer ID', axis=1, inplace=True)

    # Display correlation as an interactive heatmap
    corr = df.corr().round(2)
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

# SQL Query tab
elif choice == "SQL Query":
    st.subheader('SQL Integration')

    # Save DataFrame to SQLite database
    conn = sqlite3.connect('customer_data.sqlite')
    try:
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
    except Exception as e:
        st.write(f"An error occurred while trying to save the DataFrame to the SQLite database: {e}")
    finally:
        # Close the connection
        conn.close()

# GPT-4 Integration tab
elif choice == "GPT-4 Integration":
    st.subheader('GPT-4 Integration')

    try:
        # Load OpenAI key from file
        # with open('key.txt', 'r') as file:
            # openai_key = file.read().strip()

        # Set OpenAI API key for GPT-4
        openai.api_key = st.secrets["OPENAI_KEY"]

        # Introduction to GPT-4 integration
        st.write("Considering the crucial interplay of age, financial readiness, and retirement planning, we invite you to delve deeper into the dynamics of securing a stable financial future. With our ongoing analysis highlighting key trends in retirement preparation, we are eager to address any inquiries or uncertainties you may have about optimizing your financial plans for a comfortable retirement. Feel free to ask any questions or seek personalized insights to ensure a seamless transition into this significant phase of your life.")

        # Example related question in smaller font
        st.markdown("*Example:* What are some effective strategies for maximizing my 401K savings as I approach retirement age?", unsafe_allow_html=True)

        # Get user input
        user_input = st.text_input("Enter your question:", value='')

        # Defining the message based on the user input
        message = [{"role": "user", "content": user_input}]

        # Call GPT-4 API
        if user_input:
            print("User Input:", user_input)  # Add this line to print the user input
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                frequency_penalty=1.0,
                max_tokens=150,
                temperature=0.5
            )
            print("GPT-4's Response:", response.choices[0].message.content)  # Add this line to print the GPT-4 response
            st.write("*GPT-4's Response:*")
            st.write(response.choices[0].message.content)
    except FileNotFoundError as e:
        st.write("Error: Key file not found. Please make sure the key file is in the correct directory.")
    except Exception as e:
        st.write(f"An error occurred: {e}")

#############################################################################################################
# Model Results tab
elif choice == "Model Results":
    st.subheader('Model Results')
    
    # Classification Report
    st.subheader('Classification Report')
    st.text(classification_report(y_test, y_pred))

    # Add business conclusion
    st.write("The predictive model's performance, as evidenced by the high precision, recall, and F1-score values, reflects its accuracy in predicting retirement readiness based on customer data. The model's high accuracy rate of 96% further confirms its reliability and effectiveness in assessing customers' retirement status.")
    
    # Confusion Matrix Plot
    st.subheader('Confusion Matrix')
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", cmap="Blues", ax=ax)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    st.pyplot(fig)

    # Business Conclusion
    st.subheader('Business Conclusion')
    st.write("In summary, our comprehensive data analysis has provided crucial insights into the dynamics of retirement preparedness, emphasizing the significant roles of age, financial stability, and effective planning. Building on these insights, our predictive model demonstrates high accuracy in assessing customers' retirement readiness, as evident from the model results and predictions. By leveraging these findings, we aim to offer tailored financial strategies and personalized guidance, ensuring a seamless and secure transition into the post-retirement phase for our valued customers.")

# Predictions tab
elif choice == "Predictions":
    st.subheader('Predictions')
    st.write("Predictions are based on random forest model.")
    # User input
    age = st.text_input("Enter Age (e.g., 45.5):", value="")
    savings = st.text_input("Enter 401K Savings (e.g., 500000.25):", value="")
    
    # Predict when both fields are filled
    if st.button("Predict Retirement Status"):
        # Making the prediction
        prediction = clf.predict([[float(age), float(savings)]])[0]
        if prediction == 1:
            st.write("Based on the provided Age and 401K Savings, the model predicts that the customer is **Retired**.")
        else:
            st.write("Based on the provided Age and 401K Savings, the model predicts that the customer is **Not Retired**.")
##############################
