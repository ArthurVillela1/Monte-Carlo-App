import streamlit as st
import numpy as np
from scipy.stats import norm
import seaborn as sn
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import math
import pandas as pd
import plotly.express as px
import statistics

st.set_page_config(layout="wide")
st.title("Monte Carlo Option Pricing")

with st.sidebar:
    st.title("📈 Monte Carlo Model")
    st.write("`Created by: Arthur Villela`")
    linkedin_url = "https://www.linkedin.com/in/arthur-villela"
    github_url ="https://github.com/ArthurVillela1"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;"><a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;"></a>', unsafe_allow_html=True)
    st.sidebar.write("--------------------------")

# Getting inputs for Monte Carlo model
with st.sidebar:
    sigma = st.sidebar.slider("Volatility", 0.01, 1.00, 0.1)
    T = st.sidebar.slider("Time to Maturity (years)", 0, 3, 1)
    drifts = st.sidebar.slider("Time Steps", 1, 756, 252)
    simulations = st.sidebar.slider("Number of simulations", 1, 100, 10)
    S = st.sidebar.number_input("Current Asset Price (S)", value=95.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    K = st.sidebar.number_input("Strike (K)", value=100.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.10, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")

final_prices = []

# Estimating price paths for the underlying asset
def montecarlo_simulations(s, k, vol, rf, t, num_drifts, num_simulations):

    dt = t/num_drifts
    prices = []
    final_list = []

    for i in range(num_simulations):
        price_list = []
        s_copy = s
        for n in range(num_drifts):
            s_copy*= math.exp((rf - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * np.random.normal(0, 1))
            price_list.append(s_copy)
        prices.append(price_list)
        final_list.append(price_list[-1])

    global final_prices
    final_prices = final_list

    fig = go.Figure()
    df = pd.DataFrame(prices).T
    df.columns = [f"Simulation {i+1}" for i in range(num_simulations)]

    fig = px.line(df)
        
    fig.update_layout(title=f'Simulated Price Paths',
                      xaxis_title='Time',
                      yaxis_title='Price') 
    return fig

# Displaying chart of simulations
montecarlo_simulations(S, K, sigma, r, T, drifts, simulations)

col1, col2 = st.columns(2)

# Pricing calls and puts through Monte Carlo methods
def montecarlo_pricing(option_type, s, k, vol, rf, t):
    if option_type == "Call":
        call_value = 0
        count = 0
        for i in range(len(final_prices)):
            intrinsic_value = max(final_prices[i] - k, 0)  
            present_value = intrinsic_value * math.exp(-rf * t)
            call_value += present_value
            count += 1
        return call_value / count  
    elif option_type == "Put":
        put_value = 0
        count = 0
        for i in range(len(final_prices)):
            intrinsic_value = max(k - final_prices[i], 0)
            present_value = intrinsic_value * math.exp(-rf * t)
            put_value += present_value
            count += 1
        return put_value / count 

# Displaying call and puts values
with col1:
    st.subheader("Call Value")
    st.title(f":blue-background[{round(montecarlo_pricing('Call', S, K, r, T, sigma), 2)}]")

with col2:
    st.subheader("Put Value")
    st.title(f":green-background[{round(montecarlo_pricing('Put', S, K, r, T, sigma), 2)}]")

st.plotly_chart(montecarlo_simulations(S, K, sigma, r, T, drifts, simulations))

