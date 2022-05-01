import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.model_selection as cv
import streamlit as st
from PIL import Image
import joblib
from sklearn.metrics import mean_squared_error
import datetime
from datetime import date
bikes = pd.read_csv("https://raw.githubusercontent.com/adityajoe/BikeSharing/main/day.csv")
st.title("Bike Sharing App - Visualization Dashboard")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/00_2141_Bicycle-sharing_systems_-_Sweden.jpg/1200px-00_2141_Bicycle-sharing_systems_-_Sweden.jpg")
st.write("""* Bike sharing systems are a new generation of traditional bike rentals where the whole process from membership,
         rental and return back has become automatic. 
         Through these systems, users are able to easily rent a 
         bike from a particular position and return back at another position. """)

st.write( """* The goal here was to build an end-to-end regression model 
           to predict the number of bike sharing users on any given day.""")
st.write("""* The problem we are trying to solve here is that if we know in which months or on which days we may have more customers, 
we can plan accordingly to accomodate everyone so that our business keeps expanding""")
st.write("""* For this project I have tried to compare three models - 
Linear Regression, Decision Tree Regressors and Random Forest Regressor. 
The dataset was collected from UCI Machine Learning repository
""")
data = st.checkbox("Show raw data collected")
if data:
    st.subheader("Raw Data Collected")
    st.dataframe(bikes)

st.header("Exploratory Data Analysis")
st.write("""Here I have tried to use box plots and pair plots to see how our count of users varies with different features. 
         For example: In which weather and  days do users like to use bike sharing and when they tend to avoid it most""")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.subheader("EDA for categorical features")
option = st.selectbox(
    'Select feature to visuazlize by: ',
    ('Season', 'Year', 'Month', 'Holiday', 'Weekday', 'Weather'))
users = st.selectbox(
    'Select the type of users',
    ('Count', 'Registered Users', 'Casual Users')
)
ax = sns.boxplot(x = option, y = users, data = bikes)
st.pyplot()
if option == "Season":
    st.write("We can see that users generally use bikes more during spring and summer rather than winter and fall")
if option == "Year":
    st.write("""We can see that the number of users increased from 2011 to 2012. 
               So more and more people are starting to use bikes as a means of Transport""" )
if option == "Month":
    st.write("""we can see that total users are much more during the months - June, July, August, September and October
    and lesser during the colder months""")
if option == "Holiday":
    st.write("""The count of users on holidays and non holidays are generally the same. 
    So, we can conclude that a large number of people are using bike sharing for work commute as well!""")
if option == "Weekday":
    st.write("""The number of users is not really changing with day of the week.""")
if option == "Weather":
    st.write("""As expected, we can see fewer number of users on days with 
    rain or snow and many more users when the sun is out!""")
st.subheader("EDA for continous features")
st.write("""Here I tried to use scatter plots to see if there was change in 
         number of users with features like temperature, humidity etc. """)
option = st.selectbox(
    'Visualize the number of users by: ',
    ("Temp", "Feeling_ temp","Humidity", "Windspeed"))
ax = sns.scatterplot(x = option, y = "Count", data = bikes)
plt.ylabel("Total Number of Users")
st.pyplot()
if option == "Temp" or option == "Feeling_ temp":
    st.write("""As this dataset is from a cold country, 
    number of users is increasing with an increase in temperature or feeling temperature""")
if option == "Humidity" or option == "Windspeed":
    st.write("""As we can see, Humidity and windspeed show no relation to number of users on any given day.""")
#datacleaning
months = np.array(bikes["Month"].unique())
bikes.dropna(inplace=True)

bikes["Year"] = bikes["Year"].replace([2012],1)
bikes["Year"] = bikes["Year"].replace([2011],0)
Season_c = pd.get_dummies(bikes['Season'],drop_first=True)
Weather_c = pd.get_dummies(bikes['Weather'],drop_first=True)
final = pd.concat([bikes, Season_c, Weather_c], axis = 1)
for month in months:
  average = final["Count"][final["Month"] == month].mean()
  final["Month"] = final["Month"].replace([month],average)
final = final.drop(["Season", "Weather","Index", "Date", "Weekday", "Weather", "Count", "Casual Users", "Registered Users"], axis = 1)
Y = bikes["Count"]
st.header("Applying different ML algorithms")
st.write("* Before training my model, I  had to deal with categorical features and text features "
         "because the computer can only understand numeric features. So I did the following steps.")
st.write("1. Replaced month with the average demand during that month.")
st.write("2. Used one hot encoding to replace features year, season and Weather")
X_train, X_test, Y_train, Y_test = cv.train_test_split(final, Y, test_size = 0.33, random_state = 5)
col1,col2,col3 = st.columns(3)

with col1:
    st.subheader("Linear Regression Model")
    model1 = joblib.load('bikesmodel1.pkl')
    Y_pred = model1.predict(X_test)
    ax = sns.scatterplot(Y_test, Y_pred)
    plt.xlabel("Actual Number of Users")
    plt.ylabel("Predicted Number of Users")
    st.pyplot()
    e_value = mean_squared_error(Y_test, Y_pred, squared=False)
    st.write("* The RMSE value is", e_value)
with col2:
    st.subheader("Decision Tree Regressor Model")
    model2 = joblib.load('bikesmodel2.pkl')
    Y_pred = model2.predict(X_test)
    ax = sns.scatterplot(Y_test, Y_pred)
    plt.xlabel("Actual Number of Users")
    plt.ylabel("Predicted Number of Users")
    st.pyplot()
    e_value = mean_squared_error(Y_test, Y_pred, squared=False)
    st.write("* The RMSE value is", e_value)
with col3:
    st.subheader("Random Forest Regressor Model")
    model3 = joblib.load('bikesmodel3.pkl')
    Y_pred = model3.predict(X_test)
    ax = sns.scatterplot(Y_test, Y_pred)
    plt.xlabel("Actual Number of Users")
    plt.ylabel("Predicted Number of Users")
    st.pyplot()
    e_value = mean_squared_error(Y_test, Y_pred, squared=False)
    st.write("* The RMSE value is", e_value)
st.caption("PS. you can click on each image to see enlarged graph")
st.write("So, we can conclude that the Random Forest Model is giving us the best results on our dataset")
st.write("")
st.write("")
st.write("")
st.write("")
st.caption("You can contact me at adityajoethomas@gmail.com")







