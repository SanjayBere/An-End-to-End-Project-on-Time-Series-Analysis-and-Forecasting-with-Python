#!/usr/bin/env python
# coding: utf-8

# ### Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. 
# 
# ### Time series forecasting is the use of a model to predict future values based on previously observed values.
# 
# ### Time series are widely used for non-stationary data, like economic, weather, stock price, and retail sales in this post. We will demonstrate different approaches for forecasting retail sales time series. 
# 
# ### Let’s get started!!!!

# # Univariate Time Series Analysis

# ### We are using Superstore sales data

# In[52]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# ###  There are several categories in the Superstore sales data, we start from time series analysis and forecasting for furniture sales.

# In[53]:


df = pd.read_excel('Sample - Superstore.xls') 
df.head()


# In[54]:


furniture = df.loc[df['Category'] == 'Furniture']


# In[55]:


furniture['Order Date'].min() , furniture['Order Date'].max()


# - We have a almost 4 years of data

# ============================================================================================================
# # Data Preprocessing

# 
# ### This step includes removing columns that we don't need, check missing values, aggregate sales by date and so on.

# In[56]:


cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols,axis=1,inplace=True)
furniture.sort_values('Order Date')
furniture.isnull().sum()


# In[57]:


type(furniture)


# In[58]:


furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()


# ### Indexing With Time series Data

# In[59]:


furniture = furniture.set_index('Order Date')
furniture.index


# - Our current datetime data can be tricky to work with, therefore, we will use the averages daily sales value for that month instead, and we are using the start of each month as the timestamp.

# In[60]:


# average the daily sales value for each month 
# use start of each month as the timestamp
y = furniture['Sales'].resample('MS').mean()


# In[61]:


# Quick pick od a certain year
y['2017':]


# # Visualizing Furniture Sales Time Series Data

# In[62]:


y.plot(figsize = (15,6)) #figsize = (width , height)
plt.show()


# - Some distinguishable patterns appear when we plot the data. The time-series has seasonality pattern, such as sales are always low at the beginning of the year and high at the end of the year. 
# - There is always an upward trend within any single year with a couple of low months in the mid of the year.
# - We can also visualize our data using a method called time-series decomposition that allows us to decompose our time series into three distinct components: trend, seasonality, and noise.

# In[63]:


from pylab import rcParams

rcParams['figure.figsize'] = 18,8
decomposition = sm.tsa.seasonal_decompose(y)
fig = decomposition.plot()
plt.show()


# - The plot above clearly shows that the sales of furniture is unstable, along with its obvious seasonality.

# # Time Series forecasting with ARIMA

# - We are going to apply one of the most commonly used method for time-series forecasting, known as ARIMA, which stands for 
# 
# - Autoregressive Integrated Moving Average.
# 
# - ARIMA models are denoted with the notation ARIMA(p, d, q). These three parameters account for seasonality, trend, and noise in data:

# In[64]:


# set the typical ranges for p, d, q
p = d = q = range(0, 2)

#take all possible combination for p, d and q
pdq = list(itertools.product(p, d, q))
print("pdq : ",pdq)
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[65]:


# Using Grid Search find the optimal set of parameters that yields the best performance
result = 0
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y, order = param, seasonal_order = param_seasonal, enforce_stationary = False,enforce_invertibility=False) 
            result = mod.fit()   
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, result.aic))
        except:
            continue
            
print("Lowest AIC Value : ",result.aic.min())


# - The above output suggests that SARIMAX(1, 1, 1)x(1, 1, 0, 12) yields the lowest AIC value of 486.56. 
# - Therefore we should consider this to be optimal option.

# ### Fitting the ARIMA model

# In[66]:


#Fitting the ARIMA model using above optimal combination of p, d, q (optimal means combination at which we got lowest AIC score)

model = sm.tsa.statespace.SARIMAX(y, order = (1, 1, 1),
                                  seasonal_order = (1, 1, 0, 12)
                                 )
result = model.fit()
print(result.summary().tables[1])


# - We should always run model diagnostics to investigate any unusual behavior.

# In[67]:


#run model diagnostic to investigate any unusual behavior
result.plot_diagnostics(figsize = (16, 8))
plt.show()


# - It is not perfect, however, our model diagnostics suggests that the model residuals are near normally distributed.

# ## Validating forecasts
# To help us understand the accuracy of our forecasts, we compare predicted sales to real sales of the time series, and we set forecasts to start at 2017–01–01 to the end of the data.

# In[68]:


prediction = result.get_prediction(start = pd.to_datetime('2017-01-01'), dynamic = False)
prediction_ci = prediction.conf_int()
prediction_ci


# In[69]:


ax = y['2014':].plot(label='observed')
prediction.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(prediction_ci.index,
                prediction_ci.iloc[:, 0],
                prediction_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()


# - The line plot is showing the observed values compared to the rolling forecast predictions. 
# 
# - Overall, our forecasts align with the true values very well, showing an upward trend starts from the beginning of the year and captured the seasonality toward the end of the year.

# ### Error Analysis
# 

# In[71]:


# Evaluation metrics are Mean Squared Error(MSE) and Root Mean Squared Error(RMSE)
y_hat = prediction.predicted_mean
y_truth = y['2017-01-01':]

mse = ((y_hat - y_truth) ** 2).mean()
rmse = np.sqrt(mse)
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 2)))


# - The MSE is a measure of the quality of an estimator — it is always non-negative, and the smaller the MSE, the closer we are to finding the line of best fit.
# 
# - Root Mean Square Error (RMSE) tells us that our model was able to forecast the average daily furniture sales in the test set within 199.99 of the real sales.

# ### Producing and visualizing forecasts

# In[72]:


# forcasting for out of sample data
pred_uc = result.get_forecast(steps = 100)
pred_ci = pred_uc.conf_int()

ax = y.plot(label = 'observed', figsize = (14, 7))
pred_uc.predicted_mean.plot(ax = ax, label = 'forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color = 'k', alpha = 0.25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')

plt.legend()
plt.show()


# ### Summary : 
# 
# - Our model clearly captured furniture sales seasonality. As we forecast further out into the future, it is natural for us to become less confident in our values. 
# 
# - This is reflected by the confidence intervals generated by our model, which grow larger as we move further out into the future.
# 
