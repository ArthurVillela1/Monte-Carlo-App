import streamlit as st
import numpy as np
from scipy.stats import norm
import seaborn as sn
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import math

# Calculating option payoff
def option_payoff(option_type, S, K):
    if option_type == "Call":
        return max(S - K, 0)
    elif option_type == "Put":
        return max(K - S, 0)

def montecarlo_simulations(option_type, S, K, sigma, r, t, num_drifts, num_simulations):
    
    dt = t/num_drifts

    x_values = np.linspace(0, t, dt)
    prices = []
    option_payoffs = []
    
    for i in num_simulations:
        price = []
        asset_price *= math.exp((r - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * np.random.normal(0, 1))
        price.append(asset_price)

    prices.append(price)

    return price_paths