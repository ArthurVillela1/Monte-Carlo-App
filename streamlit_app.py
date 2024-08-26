import streamlit as st
import numpy as np
from scipy.stats import norm
import seaborn as sn
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import math

st.set_page_config(layout="wide")
st.title("Monte Carlo Option Pricing")

with st.sidebar:
    vol = st.sidebar.slider("Volatility", 0.01, 1.00, 0.1)
    t = st.sidebar.slider("Time to Maturity (months)", 1, 36, 12)
    num_drifts = st.sidebar.slider("Time Steps", 1, 756, 252)
    num_simulations = st.sidebar.slider("Number of simulations", 1, 50, 10)
    S = st.sidebar.number_input("Current Asset Price (S)", value=50.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    K = st.sidebar.number_input("Strike (K)", value=70.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    rf = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.10, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")

# Calculating option payoff
def option_payoff(option_type, S, K):
    if option_type == "Call":
        return max(S - K, 0)
    elif option_type == "Put":
        return max(K - S, 0)

def montecarlo_simulations(S, K, vol, rf, t, num_drifts, num_simulations):
    
    dt = t/num_drifts
    prices = []
    option_payoffs = []
    
    for i in range(num_simulations):
        price = []
        S*= math.exp((rf - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * np.random.normal(0, 1))
        price.append(S)

    prices.append(price)
    values = np.linspace(0, t, dt)
    
    for n in prices:
        fig.add_trace(go.Scatter(x=values, y=prices, mode='lines', name=f'Path{n}'))

    fig = go.Figure()
    fig.update_layout(title=f'Simulated Price Paths',
                      xaxis_title=time,
                      yaxis_title='Price Paths')    

    return fig

st.plotly_chart(montecarlo_simulations(K, vol, rf, t, num_drifts, num_simulations))


