# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:46:31 2023

@author: Local User
"""

from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np

import sqlalchemy
import mysql.connector

import pmdarima as pm
import matplotlib.pyplot as plt

import seaborn as sns

start_date = "2023-01-01"
end_date = "2023-08-11"

# Grain 

Soybean = yf.download("SOYB", start = start_date, end = end_date)
Corn = yf.download("CORN", start = start_date, end = end_date)
Wheat = yf.download("WEAT", start = start_date, end = end_date)

Soybean["Adjustment Multiplier"] = Soybean["Adj Close"] / Soybean["Close"]
Corn["Adjustment Multiplier"] = Corn["Adj Close"] / Corn["Close"]
Wheat["Adjustment Multiplier"] = Wheat["Adj Close"] / Wheat["Close"]

Soybean["Adj Open"] = Soybean["Open"] * Soybean["Adjustment Multiplier"]
Corn["Adj Open"] = Corn["Open"] * Corn["Adjustment Multiplier"]
Wheat["Adj Open"] = Wheat["Open"] * Wheat["Adjustment Multiplier"]

Soybean["returns"] = ((Soybean["Adj Open"] - Soybean["Adj Close"].shift(1)) / Soybean["Adj Close"].shift(1)).fillna(0)
Corn["returns"] = ((Corn["Adj Open"] - Corn["Adj Close"].shift(1)) / Corn["Adj Close"].shift(1)).fillna(0)
Wheat["returns"] = ((Wheat["Adj Open"] - Wheat["Adj Close"].shift(1)) / Wheat["Adj Close"].shift(1)).fillna(0)

# Metals

Gold = yf.download("GLD", start = start_date, end = end_date)
Silver = yf.download("SLV", start = start_date, end = end_date)
Copper = yf.download("CPER", start = start_date, end = end_date)

Gold["Adjustment Multiplier"] = Gold["Adj Close"] / Gold["Close"]
Silver["Adjustment Multiplier"] = Silver["Adj Close"] / Silver["Close"]
Copper["Adjustment Multiplier"] = Copper["Adj Close"] / Copper["Close"]

Gold["Adj Open"] = Gold["Open"] * Gold["Adjustment Multiplier"]
Silver["Adj Open"] = Silver["Open"] * Silver["Adjustment Multiplier"]
Copper["Adj Open"] = Copper["Open"] * Copper["Adjustment Multiplier"]

Gold["returns"] = ((Gold["Adj Open"] - Gold["Adj Close"].shift(1)) / Gold["Adj Close"].shift(1)).fillna(0)
Silver["returns"] = ((Silver["Adj Open"] - Silver["Adj Close"].shift(1)) / Silver["Adj Close"].shift(1)).fillna(0)
Copper["returns"] = ((Copper["Adj Open"] - Copper["Adj Close"].shift(1)) / Copper["Adj Close"].shift(1)).fillna(0)

# Energy

Crude = yf.download("USO", start = start_date, end = end_date)
NatGas = yf.download("UNG", start = start_date, end = end_date)
Gasoline = yf.download("UGA", start = start_date, end = end_date)

Crude["Adjustment Multiplier"] = Crude["Adj Close"] / Crude["Close"]
NatGas["Adjustment Multiplier"] = NatGas["Adj Close"] / NatGas["Close"]
Gasoline["Adjustment Multiplier"] = Gasoline["Adj Close"] / Gasoline["Close"]

Crude["Adj Open"] = Crude["Open"] * Crude["Adjustment Multiplier"]
NatGas["Adj Open"] = NatGas["Open"] * NatGas["Adjustment Multiplier"]
Gasoline["Adj Open"] = Gasoline["Open"] * Gasoline["Adjustment Multiplier"]

Crude["returns"] = ((Crude["Adj Open"] - Crude["Adj Close"].shift(1)) / Crude["Adj Close"].shift(1)).fillna(0)
NatGas["returns"] = ((NatGas["Adj Open"] - NatGas["Adj Close"].shift(1)) / NatGas["Adj Close"].shift(1)).fillna(0)
Gasoline["returns"] = ((Gasoline["Adj Open"] - Gasoline["Adj Close"].shift(1)) / Gasoline["Adj Close"].shift(1)).fillna(0)

# Financials

TenYearNote = yf.download("IEF", start = start_date, end = end_date)
TwoYearNote = yf.download("SHY", start = start_date, end = end_date)

TenYearNote["Adjustment Multiplier"] = TenYearNote["Adj Close"] / TenYearNote["Close"]
TwoYearNote["Adjustment Multiplier"] = TwoYearNote["Adj Close"] / TwoYearNote["Close"]

TenYearNote["Adj Open"] = TenYearNote["Open"] * TenYearNote["Adjustment Multiplier"]
TwoYearNote["Adj Open"] = TwoYearNote["Open"] * TwoYearNote["Adjustment Multiplier"]

TenYearNote["returns"] = ((TenYearNote["Adj Open"] - TenYearNote["Adj Close"].shift(1)) / TenYearNote["Adj Close"].shift(1)).fillna(0)
TwoYearNote["returns"] = ((TwoYearNote["Adj Open"] - TwoYearNote["Adj Close"].shift(1)) / TwoYearNote["Adj Close"].shift(1)).fillna(0)

# Indices

All_Futures = pd.concat([Soybean.add_prefix("Soybean_"), Corn.add_prefix("Corn_"), Wheat.add_prefix("Wheat_"),
                         Gold.add_prefix("Gold_"), Silver.add_prefix("Silver_"), Copper.add_prefix("Copper_"),
                         Crude.add_prefix("Crude_"), NatGas.add_prefix("NatGas_"), Gasoline.add_prefix("Gasoline_"),
                         TenYearNote.add_prefix("10YearNote_"), TwoYearNote.add_prefix("2YearNote_")], axis = 1)

All_Futures = All_Futures.tail(len(All_Futures) - 1)

All_Futures_Returns = All_Futures[["Soybean_returns", "Corn_returns", "Wheat_returns",
                                   "Gold_returns","Silver_returns","Copper_returns",
                                   "Crude_returns","NatGas_returns","Gasoline_returns",
                                   "10YearNote_returns","2YearNote_returns"]].copy()

Asset_Returns = All_Futures_Returns.columns

### - Begin Calculation

postive_forecast = []
positive_actual = []

negative_forecast = []
negative_actual = []

positive_pnls = []
negative_pnls = []

chunk_start = 0
chunk_end = 30

for day in range(0, len(All_Futures_Returns) - 1):
    
    Training_Set = All_Futures_Returns[chunk_start:chunk_end]
    
    if Training_Set.tail(1).index == All_Futures_Returns.tail(1).index:
        break
    
    Asset_Correlations = []
    
    for asset in Asset_Returns:
    
        
        # Isolate the regular daily returns of the first asset
        Asset_1 = Training_Set[asset]
        
        # Get the *next* day's return of all the other assets
        Other_Assets_Shifted = Training_Set.drop(asset, axis = 1).shift(-1).copy()
        
        # Combine the regular returns with the shifted returns
        Asset_1_Future_Returns = pd.concat([Asset_1, Other_Assets_Shifted], axis = 1).dropna()
        
        # Create a correlation matrix of the combined returns
        Asset_Correlation_Matrix = round(Asset_1_Future_Returns.corr(), 2)
        
        # Only the first asset is the original, so we only include the next day correlation's for that first asset
        Correlated_to_Asset_1 = Asset_Correlation_Matrix.head(1)
        
        # Append to the list and repeat for all assets
        Asset_Correlations.append(Correlated_to_Asset_1)
    
    
    All_Shifted_Future_Returns_Matrix = pd.concat(Asset_Correlations)
    
    ########################

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8), dpi = 800)
    sns.heatmap(All_Shifted_Future_Returns_Matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)

    # Set the axis labels
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title("30-Day Futures Correlation Heatmap")

    # Display the heatmap
    plt.show()

    ########################
    
    Unique_Assets = All_Shifted_Future_Returns_Matrix[All_Shifted_Future_Returns_Matrix < 1]
    
    Most_Positively_Correlated = Unique_Assets[Unique_Assets == Unique_Assets.max().sort_values(ascending = False)[0]].stack().reset_index()
    Most_Negatively_Correlated = Unique_Assets[Unique_Assets == Unique_Assets.min().sort_values(ascending = True)[0]].stack().reset_index()
    
    
    # Positively Correlated
    # First, get the 2 most positively correlated assets, then for the second asset, shift the returns
    
    Positive_Asset_1 = Training_Set[Most_Positively_Correlated["level_0"].iloc[0]]
    Positive_Asset_2_Shifted = Training_Set[Most_Positively_Correlated["level_1"].iloc[0]].shift(-1)
    
    Positive_Shifted_Returns = round(pd.concat([Positive_Asset_1, Positive_Asset_2_Shifted], axis = 1) *100, 2)
    
    # Positive Predictions
    # Assign the training data of the 2 assets.
    
    Positive_X = Positive_Shifted_Returns[[Most_Positively_Correlated["level_0"].iloc[0]]].head(len(Positive_Shifted_Returns)-1)
    Positive_Y = Positive_Shifted_Returns[Most_Positively_Correlated["level_1"].iloc[0]].head(len(Positive_Shifted_Returns)-1)
    
    Positively_Correlated_Model = pm.arima.auto_arima(  X=Positive_X, y=Positive_Y,
                                                        start_p=2, d=None,
                                                        start_q=2, max_p=5,
                                                        max_d=2, max_q=5,
                                                        start_P=1, D=None,
                                                        start_Q=1, max_P=2,
                                                        max_D=1, max_Q=2,
                                                        max_order=5, m=1,
                                                        seasonal=True, stationary=False,
                                                        information_criterion='aic', alpha=0.05,
                                                        test='kpss', seasonal_test='ocsb',
                                                        stepwise=True, n_jobs=1,
                                                        start_params=None, trend=None,
                                                        method='lbfgs', maxiter=50,
                                                        offset_test_args=None, seasonal_test_args=None,
                                                        suppress_warnings=True, error_action='trace',
                                                        trace=False, random=False,
                                                        random_state=None, n_fits=10,
                                                        return_valid_fits=False, out_of_sample_size=0,
                                                        scoring='mse', scoring_args=None,
                                                        with_intercept='auto', sarimax_kwargs=None)
        
    Positively_Correlated_Production_Data = Positive_Shifted_Returns[[Most_Positively_Correlated["level_0"].iloc[0]]].tail(1)
    
    # Get the prediction for the next day's open return
    Positive_Daily_Prediction = Positively_Correlated_Model.predict(X = Positively_Correlated_Production_Data, n_periods = 1).iloc[0]
    
    # Get the original return on the next day
    Positive_Actual_Result = round(All_Futures_Returns[Most_Positively_Correlated["level_1"].iloc[0]][chunk_start:chunk_end+1].tail(1).iloc[0]* 100, 2)
    
    postive_forecast.append(Positive_Daily_Prediction)
    positive_actual.append(Positive_Actual_Result)
    
    if (Positive_Daily_Prediction < 0) and (Positive_Actual_Result < 0):
        
        positive_pnls.append(Positive_Actual_Result)
        
    elif (Positive_Daily_Prediction > 0) and (Positive_Actual_Result > 0):
        
        positive_pnls.append(Positive_Actual_Result)
        
    else:
        
        positive_pnls.append(abs(Positive_Actual_Result)*-1)
    
    # Negatively Correlated
    # First, get the 2 most negatively correlated assets, then for the second asset, shift the returns
    
    Negative_Asset_1 = Training_Set[Most_Negatively_Correlated["level_0"].iloc[0]]
    Negative_Asset_2_Shifted = Training_Set[Most_Negatively_Correlated["level_1"].iloc[0]].shift(-1)
    
    Negative_Shifted_Returns = round(pd.concat([Negative_Asset_1, Negative_Asset_2_Shifted], axis = 1) *100, 2)
    
    # Negative Predictions
    # Assign the training data of the 2 assets.
    
    Negative_X = Negative_Shifted_Returns[[Most_Negatively_Correlated["level_0"].iloc[0]]].head(len(Negative_Shifted_Returns)-1)
    Negative_Y = Negative_Shifted_Returns[Most_Negatively_Correlated["level_1"].iloc[0]].head(len(Negative_Shifted_Returns)-1)
    
    Negatively_Correlated_Model = pm.arima.auto_arima(  X=Negative_X, y=Negative_Y,
                                                        start_p=2, d=None,
                                                        start_q=2, max_p=5,
                                                        max_d=2, max_q=5,
                                                        start_P=1, D=None,
                                                        start_Q=1, max_P=2,
                                                        max_D=1, max_Q=2,
                                                        max_order=5, m=1,
                                                        seasonal=True, stationary=False,
                                                        information_criterion='aic', alpha=0.05,
                                                        test='kpss', seasonal_test='ocsb',
                                                        stepwise=True, n_jobs=1,
                                                        start_params=None, trend=None,
                                                        method='lbfgs', maxiter=50,
                                                        offset_test_args=None, seasonal_test_args=None,
                                                        suppress_warnings=True, error_action='trace',
                                                        trace=False, random=False,
                                                        random_state=None, n_fits=10,
                                                        return_valid_fits=False, out_of_sample_size=0,
                                                        scoring='mse', scoring_args=None,
                                                        with_intercept='auto', sarimax_kwargs=None)
        
    Negatively_Correlated_Production_Data = Negative_Shifted_Returns[[Most_Negatively_Correlated["level_0"].iloc[0]]].tail(1)
    
    # Get the prediction for the next day's open return
    Negative_Daily_Prediction = Negatively_Correlated_Model.predict(X = Negatively_Correlated_Production_Data, n_periods = 1).iloc[0]
    
    # Get the original return on the next day
    Negative_Actual_Result = round(All_Futures_Returns[Most_Negatively_Correlated["level_1"].iloc[0]][chunk_start:chunk_end+1].tail(1).iloc[0]*100,2)
    
    negative_forecast.append(Negative_Daily_Prediction)
    negative_actual.append(Negative_Actual_Result)
    
    if (Negative_Daily_Prediction < 0) and (Negative_Actual_Result < 0):
        
        negative_pnls.append(Negative_Actual_Result)
        
    elif (Negative_Daily_Prediction > 0) and (Negative_Actual_Result > 0):
        
        negative_pnls.append(Negative_Actual_Result)
        
    else:
        
        negative_pnls.append(abs(Negative_Actual_Result)*-1)
    
    chunk_start = chunk_start + 1
    chunk_end = chunk_end + 1


pnl_dataframe = pd.DataFrame({"positive_correlated_pair":positive_pnls, "negative_correlated_pair":negative_pnls})

pnl_dataframe["positive_correlated_cum_returns"] = pnl_dataframe["positive_correlated_pair"].cumsum()
pnl_dataframe["negative_correlated_cum_returns"] = pnl_dataframe["negative_correlated_pair"].cumsum()

plt.figure(figsize=(10, 8), dpi = 800)

plt.plot(pnl_dataframe["positive_correlated_cum_returns"])
plt.plot(pnl_dataframe["negative_correlated_cum_returns"])

plt.legend(["Returns of Positively Correlated Pairs", "Returns of Negatively Correlated Pairs"])
plt.title("Total Strategy Performance (Before Fees) 01-01-23 to 08-11-23")

plt.show()