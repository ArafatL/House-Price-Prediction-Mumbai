#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[6]:


data = pd.read_csv("D:\Project Dissertation\Mumbai_House_Data.csv")


# In[7]:


data.head()


# In[8]:


data.shape


# In[9]:


data.info()


# In[7]:


for column in data.columns:
    print(data[column].value_counts())
    print("*"*20)


# In[12]:


data.isna().sum()


# In[13]:


data.describe()


# In[14]:


data.info()


# In[15]:


data['Area'].value_counts()


# In[16]:


data['Location'].value_counts()


# In[17]:


data.shape


# In[18]:


X = data.drop(columns=['Price'])
y = data['Price']


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1)


# In[21]:


print(X_train.shape)
print(X_test.shape)


# ## Applying Linear Regression

# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Handling categorical data (One-hot encoding for 'Location' column)
ohe = OneHotEncoder()
location_encoded = ohe.fit_transform(data[['Location']]).toarray()

# Creating a DataFrame with the encoded location
location_encoded_df = pd.DataFrame(location_encoded, columns=ohe.get_feature_names(['Location']))

# Concatenating the original data with the encoded location DataFrame
data_encoded = pd.concat([data.drop('Location', axis=1), location_encoded_df], axis=1)

# Splitting the dataset into independent (X) and dependent (y) variables
X = data_encoded.drop('Price', axis=1)
y = data_encoded['Price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Training the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predicting the prices for the testing set
y_pred = lr_model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

mae, rmse, r2


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns

# Analyzing the distribution of 'Price' and other numerical features
numerical_features = ['Price', 'Area', 'No. of Bedrooms']

# Plotting distributions
plt.figure(figsize=(15, 5))
for i, feature in enumerate(numerical_features):
    plt.subplot(1, 3, i+1)
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Displaying statistical summary of numerical features
data[numerical_features].describe()


# In[33]:


from sklearn.preprocessing import StandardScaler

# Capping outliers at the 95th percentile for 'Price' and 'Area'
for feature in ['Price', 'Area']:
    cap_value = data[feature].quantile(0.95)
    data[feature] = data[feature].clip(upper=cap_value)

# Re-encoding the 'Location' column (as the dataset has been modified)
location_encoded = ohe.transform(data[['Location']]).toarray()
location_encoded_df = pd.DataFrame(location_encoded, columns=ohe.get_feature_names(['Location']))


# Re-creating the encoded dataset
data_encoded = pd.concat([data.drop('Location', axis=1), location_encoded_df], axis=1)

# Splitting the dataset into independent (X) and dependent (y) variables again
X = data_encoded.drop('Price', axis=1)
y = data_encoded['Price']

# Scaling the numerical features (excluding one-hot encoded features)
scaler = StandardScaler()
numerical_features.remove('Price')  # Remove 'Price' as it's the target variable
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Splitting the data into training and testing sets again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Re-training the Linear Regression model
lr_model.fit(X_train, y_train)

# Re-predicting and re-evaluating the model
y_pred = lr_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

mae, rmse, r2


# In[32]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3) # Line of perfect predictions
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()


# In[36]:


# Creating a new DataFrame for the input data with the same structure as the training data
input_data_reconstructed = pd.DataFrame(columns=X_train.columns)

# Filling in the input data with provided values and defaults
input_data_reconstructed.loc[0, :] = 0  # Initializing all columns to 0
input_data_reconstructed['Area'] = 500
input_data_reconstructed['No. of Bedrooms'] = 2

# Setting the one-hot encoded location
location_column = 'Location_' + 'Marine Lines'
if location_column in input_data_reconstructed.columns:
    input_data_reconstructed[location_column] = 1
else:
    # If the location is not in the training set, we can't make a prediction for it
    raise ValueError("The specified location is not available in the training set for prediction.")

# Scaling the numerical features
input_data_reconstructed[numerical_features] = scaler.transform(input_data_reconstructed[numerical_features])

# Making the prediction
predicted_price_final = lr_model.predict(input_data_reconstructed)
predicted_price_final[0]


# ## Applying Random Forest Model

# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Dropping the unnecessary 'Unnamed: 0' column
data = data.drop(columns=['Unnamed: 0'])

# Encoding categorical data ('Location')
encoder = OneHotEncoder(sparse=False)
location_encoded = encoder.fit_transform(data[['Location']])

# Creating a DataFrame from the encoded features
location_encoded_df = pd.DataFrame(location_encoded, 
                                   columns=[f"Location_{cat}" for cat in encoder.categories_[0]])

# Dropping the original 'Location' column and adding the encoded features
data = data.drop(columns=['Location'])
data_encoded = pd.concat([data, location_encoded_df], axis=1)

# Splitting the dataset into features (X) and target variable (y)
X = data_encoded.drop('Price', axis=1)
y = data_encoded['Price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

rmse, r2


# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns

# Setting up the plot
plt.figure(figsize=(10, 6))

# Scatter plot of Actual vs Predicted values
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Diagonal line

plt.show()


# In[39]:


# Creating a new data point for prediction
# We'll use default values for other features
# 'New/Resale' is set to 0 (New) and all amenities are set to 0 (not available) as default

# Encoding the 'Location' for the new data point
location_for_prediction = encoder.transform([['Marine Lines']])

# Creating a DataFrame for the new data point
new_data_point = pd.DataFrame({
    'Area': [600],
    'No. of Bedrooms': [3],
    'New/Resale': [0],  # Assuming 'New'
    'Gymnasium': [0],
    'Lift Available': [0],
    'Car Parking': [0],
    'Maintenance Staff': [0],
    '24x7 Security': [0],
    'Children\'s Play Area': [0],
    'Clubhouse': [0],
    'Intercom': [0],
    'Landscaped Gardens': [0],
    'Indoor Games': [0],
    'Gas Connection': [0],
    'Jogging Track': [0],
    'Swimming Pool': [0]
})

# Adding the encoded location to the new data point
for i, col in enumerate(encoder.categories_[0]):
    new_data_point[f'Location_{col}'] = location_for_prediction[0][i]

# Ensuring the new data point has the same features as the training data
new_data_point = new_data_point.reindex(columns=X.columns, fill_value=0)

# Predicting the price
predicted_price = rf_model.predict(new_data_point)

predicted_price[0]


# ## Applying Support Vector Machine

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Support Vector Regressor
svm_model = SVR()

# Train the model
svm_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluate the model
rmse_svm = mean_squared_error(y_test, y_pred_svm, squared=False)
r2_svm = r2_score(y_test, y_pred_svm)

rmse_svm, r2_svm


# In[ ]:


from sklearn.feature_selection import RFE

# Create the RFE object and rank each feature
svm_rfe = SVR(kernel='linear')  # Using a linear kernel for feature importance
rfe = RFE(estimator=svm_rfe, n_features_to_select=10)  # Selecting top 10 features
rfe = rfe.fit(X_train_scaled, y_train)

# Summarize the selection of the attributes
selected_features = pd.DataFrame({'Feature':X.columns, 'Ranking':rfe.ranking_, 'Selected':rfe.support_})
selected_features = selected_features.sort_values(by="Ranking")
selected_features.head(10)  # Displaying the top 10 features


# In[ ]:


# Calculating the correlation matrix
correlation_matrix = data_encoded.corr()

# Correlation of features with the target variable 'Price'
correlation_with_target = correlation_matrix['Price'].sort_values(ascending=False)

# Displaying the top correlated features
correlation_with_target.head(11)  # Top 10 features along with the target variable itself


# In[ ]:


# Selecting the top 10 features (excluding the target variable itself)
top_features = correlation_with_target.index[1:11]

# Updating the training and testing sets to include only the top features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Standardizing the selected features
scaler_selected = StandardScaler()
X_train_selected_scaled = scaler_selected.fit_transform(X_train_selected)
X_test_selected_scaled = scaler_selected.transform(X_test_selected)

# Retrain the SVM model
svm_model_selected = SVR()
svm_model_selected.fit(X_train_selected_scaled, y_train)

# Predicting on the test set
y_pred_svm_selected = svm_model_selected.predict(X_test_selected_scaled)

# Evaluate the model
rmse_svm_selected = mean_squared_error(y_test, y_pred_svm_selected, squared=False)
r2_svm_selected = r2_score(y_test, y_pred_svm_selected)

rmse_svm_selected, r2_svm_selected


# In[ ]:


# Setting up the plot
plt.figure(figsize=(10, 6))

# Scatter plot of Actual vs Predicted values for SVM model
sns.scatterplot(x=y_test, y=y_pred_svm_selected)
plt.title('Actual vs Predicted Prices (SVM with Selected Features)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Diagonal line

plt.show()


# In[ ]:


# Creating a new data point for prediction with the same details: Area - 600, Location - Marine Lines, No. of Bedrooms - 3
new_data_point_svm = pd.DataFrame({
    'Area': [600],
    'No. of Bedrooms': [3]
})

# Adding default values (0) for the location features
for feature in top_features:
    if feature.startswith('Location_'):
        new_data_point_svm[feature] = 0

# Set the value for 'Location_Marine Lines' if it's among the selected features
if 'Location_Marine Lines' in top_features:
    new_data_point_svm['Location_Marine Lines'] = 1

# Ensure the new data point has the same features as the training data
new_data_point_svm = new_data_point_svm.reindex(columns=X_train_selected.columns, fill_value=0)

# Standardizing the new data point
new_data_point_svm_scaled = scaler_selected.transform(new_data_point_svm)

# Predicting the price
predicted_price_svm = svm_model_selected.predict(new_data_point_svm_scaled)

predicted_price_svm[0]


# # Comapring The Three mMdels and Identifying the Best Model for Prediction

# In[ ]:


from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the model on the training set
lr_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate the model
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
r2_lr = r2_score(y_test, y_pred_lr)

# Summarizing the results of all three models
model_results = {
    'Model': ['Linear Regression', 'Random Forest', 'SVM'],
    'RMSE': [rmse_lr, rmse, rmse_svm_selected],
    'R-Squared': [r2_lr, r2, r2_svm_selected]
}

model_comparison = pd.DataFrame(model_results)
model_comparison


# In[ ]:


pip install ipywidgets


# In[ ]:


import ipywidgets as widgets
from IPython.display import display

# Function to make predictions
def predict_price(area, bedrooms, new_resale, gymnasium, lift, parking, maintenance_staff, security, 
                  play_area, clubhouse, intercom, gardens, indoor_games, gas_connection, jogging_track, pool):
    # Create a data point from the inputs
    input_data = pd.DataFrame({
        'Area': [area],
        'No. of Bedrooms': [bedrooms],
        'New/Resale': [new_resale],
        'Gymnasium': [gymnasium],
        'Lift Available': [lift],
        'Car Parking': [parking],
        'Maintenance Staff': [maintenance_staff],
        '24x7 Security': [security],
        'Children\'s Play Area': [play_area],
        'Clubhouse': [clubhouse],
        'Intercom': [intercom],
        'Landscaped Gardens': [gardens],
        'Indoor Games': [indoor_games],
        'Gas Connection': [gas_connection],
        'Jogging Track': [jogging_track],
        'Swimming Pool': [pool]
    })

    # Ensuring the input data has the same feature columns as the training data
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    
    # Predicting the price
    predicted_price = rf_model.predict(input_data)
    print(f"Predicted Price: ₹{predicted_price[0]:,.2f}")

# Creating widgets
area_widget = widgets.IntSlider(value=600, min=200, max=10000, step=50, description='Area (sq.ft):')
bedrooms_widget = widgets.IntSlider(value=2, min=1, max=10, step=1, description='Bedrooms:')
new_resale_widget = widgets.Dropdown(options=[('New', 0), ('Resale', 1)], description='New/Resale:')
gymnasium_widget = widgets.Checkbox(value=False, description='Gymnasium')
lift_widget = widgets.Checkbox(value=False, description='Lift Available')
parking_widget = widgets.Checkbox(value=False, description='Car Parking')
maintenance_staff_widget = widgets.Checkbox(value=False, description='Maintenance Staff')
security_widget = widgets.Checkbox(value=False, description='24x7 Security')
play_area_widget = widgets.Checkbox(value=False, description="Children's Play Area")
clubhouse_widget = widgets.Checkbox(value=False, description='Clubhouse')
intercom_widget = widgets.Checkbox(value=False, description='Intercom')
gardens_widget = widgets.Checkbox(value=False, description='Landscaped Gardens')
indoor_games_widget = widgets.Checkbox(value=False, description='Indoor Games')
gas_connection_widget = widgets.Checkbox(value=False, description='Gas Connection')
jogging_track_widget = widgets.Checkbox(value=False, description='Jogging Track')
pool_widget = widgets.Checkbox(value=False, description='Swimming Pool')

# Button to make predictions
predict_button = widgets.Button(description="Predict Price")

# Arranging widgets in the interface
input_widgets = widgets.VBox([area_widget, bedrooms_widget, new_resale_widget, gymnasium_widget, lift_widget,
                              parking_widget, maintenance_staff_widget, security_widget, play_area_widget,
                              clubhouse_widget, intercom_widget, gardens_widget, indoor_games_widget,
                              gas_connection_widget, jogging_track_widget, pool_widget, predict_button])

# Event handler for the prediction button
def on_predict_button_clicked(b):
    predict_price(area_widget.value, bedrooms_widget.value, new_resale_widget.value, gymnasium_widget.value, 
                  lift_widget.value, parking_widget.value, maintenance_staff_widget.value, security_widget.value, 
                  play_area_widget.value, clubhouse_widget.value, intercom_widget.value, gardens_widget.value, 
                  indoor_games_widget.value, gas_connection_widget.value, jogging_track_widget.value, pool_widget.value)

predict_button.on_click(on_predict_button_clicked)

# Display the interface
display(input_widgets)


# In[ ]:


import ipywidgets as widgets
from IPython.display import display

# Define the prediction function
def predict_price(area, bedrooms, ...):  # include all necessary parameters
    # Prepare the input data and make a prediction using the trained model
    # ...

# Create widgets for inputs
area_widget = widgets.IntSlider(min=200, max=10000, step=50, value=600, description='Area')
# ... create other widgets ...

# Create a button to trigger the prediction
predict_button = widgets.Button(description='Predict Price')

# Define what happens when the button is clicked
def on_predict_button_clicked(b):
    # Call the predict function with the widget values
    predict_price(area_widget.value, ...)

predict_button.on_click(on_predict_button_clicked)

# Display the widgets
display(area_widget, ..., predict_button)


# In[ ]:


pip install streamlit


# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Function to load the trained model and other necessary objects
def load_model():
    with open('rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    # Load other objects like scaler, encoder if used in your model
    return model

# UI layout
def main():
    st.title("House Price Prediction")

    # Create UI elements to accept user input for each feature
    area = st.slider('Area (sq.ft)', min_value=200, max_value=10000, value=600, step=50)
    bedrooms = st.slider('Number of Bedrooms', 1, 10, 2)
    # Add other input features in a similar way

    # Button to make predictions
    if st.button('Predict Price'):
        # Prepare the input data in the format your model expects
        input_data = pd.DataFrame([[area, bedrooms, ...]])  # Add all features
        # Scale/encode the input data if necessary

        # Load the model
        model = load_model()

        # Make prediction
        prediction = model.predict(input_data)
        st.write(f"The predicted price is: ₹{prediction[0]:,.2f}")

if __name__ == '__main__':
    main()


# In[ ]:


streamlit run app.py


# In[ ]:




