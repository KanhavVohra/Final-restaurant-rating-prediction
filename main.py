import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

# Set the title
st.title('Zomato Restaurants Rating Prediction')
tab_titles = [ 
    "Overview Of Data",
    "Data Analysis",
    "Machine Learning Model",
    
]
tabs = st.tabs(tab_titles)

with tabs[0]:
    st.title('Problem Statement')
    st.markdown('The primary objective of this project is to conduct a comprehensive Exploratory Data Analysis (EDA) on a given dataset and develop an optimized Machine Learning Model. The model aims to assist restaurants in predicting their respective ratings by considering multiple relevant features. Through thorough EDA, the dataset is carefully analyzed to gain valuable insights. Subsequently, a suitable Machine Learning Model is built, which effectively leverages these insights to accurately predict restaurant ratings. The optimized model provides a valuable tool for restaurants to make informed decisions based on various factors impacting their ratings.')

    df_data_desctription = pd.read_csv('data description.csv')
    st.title('Data Description')
    st.markdown('The data for the "Zomato Bangalore Restaurants" dataset was gathered from the Zomato platform, a well-known online service that facilitates food delivery and provides information about restaurants. Zomato collects data from various sources, including user reviews, restaurant menus, and official listings. The dataset offers insights into the restaurant landscape in Bangalore, enabling analysis and exploration of various factors influencing dining choices.')
    st.table(df_data_desctription)

    st.title('Proposed Solution')
    st.markdown('EDA(Exploratory Data Analysis) to find the relation between different attributes, a machine-learning algorithm to predict the rating and a Web Application(Streamlit) to feed in user input. The client will have to fill in the required features as input which will lead to the required results. The features then will be passed into the backend where they will be validated and preprocessed and then it will be passed to a hyperparameter-tuned machine learning model to predict the final outcome. ')


with tabs[1]:
    df_analysis = pd.read_csv('zom_transfrom_before.csv')
    st.title('Cleaned Dataset')
    st.markdown('Some columns were removed from the dataset as they were deemed unnecessary for model building. The cleaned dataset, which includes only relevant columns, is being used for both model analysis and further analysis.')
    st.table(df_analysis.head())

    # Center align the title
    st.markdown("<h1 style='text-align: center;'>Top 10 cuisines</h1>", unsafe_allow_html=True)

   # Get the top 10 cuisines based on their counts
    cuisines = df_analysis['cuisines'].value_counts()[:10]

    # Create the bar plot using Plotly
    fig_cuisines = px.bar(x=cuisines.index, y=cuisines.values, color=cuisines.index)

    # Set the plot title and axis labels
    fig_cuisines.update_layout(
        xaxis_title='Cuisines',
        yaxis_title='Count',
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40)  # Adjust the margins for centering the plot
    )

    st.plotly_chart(fig_cuisines, use_container_width=True)

    st.markdown("<h1 style='text-align: center;'>Top 10 Locations</h1>", unsafe_allow_html=True)

    
    # Get the top 10 locations based on their counts
    top_locations = df_analysis['location'].value_counts().head(10)

    # Create the bar plot using Plotly
    fig_locations = px.bar(x=top_locations.index, y=top_locations.values)

    # Set the plot title and axis labels
    fig_locations.update_layout(
        xaxis_title='Locations',
        yaxis_title='Count',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Render the bar plot using Streamlit
    st.plotly_chart(fig_locations,use_container_width=True)

    st.markdown("<h1 style='text-align: center;'>Book Table vs Online Order</h1>", unsafe_allow_html=True)


    # Calculate the counts for "book_table" and "online_order"
    book_table_counts = df_analysis['book_table'].value_counts()
    online_order_counts = df_analysis['online_order'].value_counts()

    # Create subplots using Plotly
    fig_subplots = sp.make_subplots(rows=1, cols=2, subplot_titles=("Book Table", "Online Order"))

    # Add bar plots to subplots
    fig_subplots.add_trace(go.Bar(x=book_table_counts.index, y=book_table_counts.values), row=1, col=1)
    fig_subplots.add_trace(go.Bar(x=online_order_counts.index, y=online_order_counts.values), row=1, col=2)

    # Set titles and axis labels
    fig_subplots.update_layout(
        xaxis1_title="Book Table",
        yaxis1_title="Count",
        xaxis2_title="Online Order",
        yaxis2_title="Count"
    )
    fig_subplots.update_traces(showlegend=False)
    # Render the subplot using Streamlit
    st.plotly_chart(fig_subplots, use_container_width=True)


    st.markdown("<h1 style='text-align: center;'>Cost Analysis</h1>", unsafe_allow_html=True)

  # Calculate the maximum, minimum, and mean values of the "cost" column
    cost_max = df_analysis['cost'].max()
    cost_min = df_analysis['cost'].min()
    cost_mean = df_analysis['cost'].mean()

    # Define colors for the bars based on increasing cost
    colors = ['green', 'blue', 'orange']

    # Create a bar plot using Plotly
    fig_cost_analysis = go.Figure(data=[
        go.Bar(x=['Maximum', 'Minimum', 'Mean'],
            y=[cost_max, cost_min, cost_mean],
            marker=dict(color=colors))
    ])

    # Set the plot title and axis labels
    fig_cost_analysis.update_layout(
                    xaxis_title='Statistics',
                    yaxis_title='Cost')

    # Render the bar plot using Streamlit
    st.plotly_chart(fig_cost_analysis,use_container_width=True)



with tabs[2]:
    # Load the trained regressor model
    regressor = pickle.load(open('ExtraTreesRegressor.pkl', 'rb'))

    # Load the dataframe
    df = pd.read_csv('zom_transfrom_before.csv')

    # Create dictionaries for columns: city, rest_type, cuisines
    location_dict = {city: i for i, city in enumerate(df['location'].unique())}
    rest_type_dict = {rest_type: i for i, rest_type in enumerate(df['rest_type'].unique())}
    cuisines_dict = {cuisine: i for i, cuisine in enumerate(df['cuisines'].unique())}



    # Selectbox for Online Order
    online_order_dict = {'Yes': 1, 'No': 0}
    online_order = st.selectbox('Online Order', ('Yes', 'No'))
    online_order = online_order_dict[online_order]

    # Selectbox for Book Table
    book_table_dict = {'Yes': 1, 'No': 0}
    book_table = st.selectbox('Book Table', ('Yes', 'No'))
    book_table = book_table_dict[book_table]

    # Selectbox for City
    location = st.selectbox('Location', list(location_dict.keys()))

    # Selectbox for Restaurant Type
    restaurant_type = st.selectbox('Restaurant Type', list(rest_type_dict.keys()))

    # Selectbox for Cuisines
    cuisines = st.selectbox('Cuisines', list(cuisines_dict.keys()))

    # Number input for Votes
    votes = st.number_input('Votes', value=0)

    # Number input for Cost
    cost = st.number_input('Cost', value=0)

    # Prepare the input data as a DataFrame
    query = pd.DataFrame({
        'online_order': [online_order],
        'book_table': [book_table],
        'location': [location_dict[location]],
        'restaurant_type': [rest_type_dict[restaurant_type]],
        'cuisines': [cuisines_dict[cuisines]],
        'votes': [votes],
        'cost': [cost]
    })

    # Predict the rating
    if st.button('Predict'):
        if votes == 0 or cost == 0:
            st.error("Cannot predict rating when either Votes or Cost is zero.")
        else:
            # Make the prediction
            rating_prediction = regressor.predict(query.values)

            # Round the prediction to 2 decimal places
            rounded_prediction = np.round(rating_prediction[0], 2)  # Extract the value and round

            # Display the prediction
            st.title('The predicted rating is: ' + str(rounded_prediction))

