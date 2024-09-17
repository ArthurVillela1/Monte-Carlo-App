import streamlit as st
import numpy as np
import plotly.graph_objs as go
import math
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Monte Carlo Option Pricing with Greeks")

with st.sidebar:
    st.title("📈 Monte Carlo Model")
    st.write("`Created by: Arthur Villela`")
    linkedin_url = "https://www.linkedin.com/in/arthur-villela"
    github_url ="https://github.com/ArthurVillela1"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;"><a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;"></a>', unsafe_allow_html=True)
    st.sidebar.write("--------------------------")

# Sidebar inputs
with st.sidebar:
    sigma = st.sidebar.slider("Volatility", 0.01, 1.00, 0.1)
    T = st.sidebar.slider("Time to Maturity (years)", 0, 3, 1)
    drifts = st.sidebar.slider("Time Steps", 1, 756, 252)
    simulations = st.sidebar.slider("Number of simulations", 1, 100, 10)
    S = st.sidebar.number_input("Current Asset Price (S)", value=95.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    K = st.sidebar.number_input("Strike (K)", value=100.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.10, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")

final_prices = []

# Monte Carlo simulations for underlying asset
def montecarlo_simulations(s, k, vol, rf, t, num_drifts, num_simulations):
    dt = t/num_drifts
    prices = []
    final_list = []

    for i in range(num_simulations):
        price_list = []
        s_copy = s
        for n in range(num_drifts):
            s_copy *= math.exp((rf - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * np.random.normal(0, 1))
            price_list.append(s_copy)
        prices.append(price_list)
        final_list.append(price_list[-1])

    global final_prices
    final_prices = final_list

    fig = go.Figure()
    df = pd.DataFrame(prices).T
    df.columns = [f"Simulation {i+1}" for i in range(num_simulations)]

    fig = px.line(df)
    fig.update_layout(title=f'Simulated Price Paths', xaxis_title='Time', yaxis_title='Price')
    return fig

# Pricing options via Monte Carlo method
def montecarlo_pricing(option_type, s, k, vol, rf, t):
    if option_type == "Call":
        call_value = 0
        for i in range(len(final_prices)):
            intrinsic_value = max(final_prices[i] - k, 0)
            present_value = intrinsic_value * math.exp(-rf * t)
            call_value += present_value
        return call_value / len(final_prices)
    elif option_type == "Put":
        put_value = 0
        for i in range(len(final_prices)):
            intrinsic_value = max(k - final_prices[i], 0)
            present_value = intrinsic_value * math.exp(-rf * t)
            put_value += present_value
        return put_value / len(final_prices)

# Greeks: Delta, Gamma, Vega, Theta, Rho for both Call and Put options
def montecarlo_greeks(s, k, vol, rf, t, option_type, epsilon_s=100.0, epsilon_vol=0.01, epsilon_rf=0.001, epsilon_t=1/365):  
    # Use separate epsilon for different variables
    if option_type == "Call":
        price_s = montecarlo_pricing("Call", s, k, vol, rf, t)

        # Delta (sensitivity to asset price changes)
        price_s_up = montecarlo_pricing("Call", s + epsilon_s, k, vol, rf, t)
        delta = (price_s_up - price_s) / epsilon_s

        # Gamma (rate of change of Delta)
        price_s_down = montecarlo_pricing("Call", s - epsilon_s, k, vol, rf, t)
        gamma = (price_s_up - 2 * price_s + price_s_down) / epsilon_s**2

        # Vega (sensitivity to volatility changes)
        price_vol_up = montecarlo_pricing("Call", s, k, vol + epsilon_vol, rf, t)
        vega = (price_vol_up - price_s) / epsilon_vol

        # Theta (sensitivity to time decay)
        price_t_down = montecarlo_pricing("Call", s, k, vol, rf, t - epsilon_t)
        theta = (price_t_down - price_s) / epsilon_t

        # Rho (sensitivity to interest rate changes)
        price_rf_up = montecarlo_pricing("Call", s, k, vol, rf + epsilon_rf, t)
        rho = (price_rf_up - price_s) / epsilon_rf

    elif option_type == "Put":
        price_s = montecarlo_pricing("Put", s, k, vol, rf, t)

        # Delta (Put)
        price_s_up = montecarlo_pricing("Put", s + epsilon_s, k, vol, rf, t)
        delta = (price_s_up - price_s) / epsilon_s

        # Gamma (Put)
        price_s_down = montecarlo_pricing("Put", s - epsilon_s, k, vol, rf, t)
        gamma = (price_s_up - 2 * price_s + price_s_down) / epsilon_s**2

        # Vega (Put)
        price_vol_up = montecarlo_pricing("Put", s, k, vol + epsilon_vol, rf, t)
        vega = (price_vol_up - price_s) / epsilon_vol

        # Theta (Put)
        price_t_down = montecarlo_pricing("Put", s, k, vol, rf, t - epsilon_t)
        theta = (price_t_down - price_s) / epsilon_t

        # Rho (Put)
        price_rf_up = montecarlo_pricing("Put", s, k, vol, rf + epsilon_rf, t)
        rho = (price_rf_up - price_s) / epsilon_rf

    return delta, gamma, vega, theta, rho

# Displaying simulations and prices
st.plotly_chart(montecarlo_simulations(S, K, sigma, r, T, drifts, simulations))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Call Value")
    st.title(f":blue-background[{round(montecarlo_pricing('Call', S, K, sigma, r, T), 2)}]")
    st.header("Call Option Greeks")
    call_delta, call_gamma, call_vega, call_theta, call_rho = montecarlo_greeks(S, K, sigma, r, T, "Call")
    st.subheader(f"**Delta (Δ):** {round(call_delta, 2)}")
    st.subheader(f"**Gamma (Γ):** {round(call_gamma, 2)}")
    st.subheader(f"**Theta (Θ):** {round(call_theta, 2)}")
    st.subheader(f"**Vega (ν):** {round(call_vega, 2)}")
    st.subheader(f"**Rho (ρ):** {round(call_rho, 2)}")

with col2:
    st.subheader("Put Value")
    st.title(f":green-background[{round(montecarlo_pricing('Put', S, K, sigma, r, T), 2)}]")
    st.header("Put Option Greeks")
    put_delta, put_gamma, put_vega, put_theta, put_rho = montecarlo_greeks(S, K, sigma, r, T, "Put")
    st.subheader(f"**Delta (Δ):** {round(put_delta, 2)}")
    st.subheader(f"**Gamma (Γ):** {round(put_gamma, 2)}")
    st.subheader(f"**Theta (Θ):** {round(put_theta, 2)}")
    st.subheader(f"**Vega (ν):** {round(put_vega, 2)}")
    st.subheader(f"**Rho (ρ):** {round(put_rho, 2)}")


