import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the trained regressor model
regressor = pickle.load(open('ExtraTreesRegressor.pkl', 'rb'))

# Load the dataframe
df = pd.read_csv('zom_transfrom_before.csv')

# Create dictionaries for columns: city, rest_type, cuisines
location_dict = {city: i for i, city in enumerate(df['location'].unique())}
rest_type_dict = {rest_type: i for i, rest_type in enumerate(df['rest_type'].unique())}
cuisines_dict = {cuisine: i for i, cuisine in enumerate(df['cuisines'].unique())}

# Set the title
st.title('Zomato Restaurants Rating')

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
    if votes == 0 and cost == 0:
        st.error("Cannot predict rating when both Votes and Cost are zero.")
    else:
        # Make the prediction
        rating_prediction = regressor.predict(query.values)

        # Round the prediction to 2 decimal places
        rounded_prediction = np.round(rating_prediction[0], 2)  # Extract the value and round

        # Display the prediction
        st.title('The predicted rating is: ' + str(rounded_prediction))
