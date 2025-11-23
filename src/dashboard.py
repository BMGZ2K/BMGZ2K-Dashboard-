import streamlit as st
import pandas as pd
import json
import time
import plotly.graph_objects as go
import os
import config

st.set_page_config(page_title="Binance Bot Dashboard", layout="wide", page_icon="🚀")

# Custom CSS for "Premium" look
st.markdown("""
<style>
    /* Global Settings */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #00FFFF !important; /* Cyan Headings */
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    
    /* Metric Containers */
    div[data-testid="stMetric"] {
        background-color: #111111;
        border: 1px solid #333333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,255,255,0.1); /* Subtle Cyan Glow */
        color: #FFFFFF;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #00FFFF !important; /* Bright Cyan for Labels */
        font-size: 16px;
        font-weight: 600;
    }
    
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important; /* Pure White for Values */
        font-size: 28px;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 16px;
        font-weight: bold;
        background-color: #222;
        padding: 2px 5px;
        border-radius: 4px;
    }
    
    p, li, span {
        color: #E0E0E0 !important; /* Bright Grey for body text */
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #333333;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #00FFFF;
        color: #000000;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #00CCCC;
        box-shadow: 0 0 10px #00FFFF;
    }
    
    /* JSON/Code Blocks */
    .stJson {
        background-color: #111111;
        color: #00FF00;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Binance Futures Bot: Profit Evolution")

# Load State
state = {}
if os.path.exists('bot_state.json'):
    try:
        with open('bot_state.json', 'r') as f:
            state = json.load(f)
    except Exception as e:
        st.error(f"Error loading state: {e}")

# --- SUMMARY DASHBOARD ---
if state:
    # Check if state is multi-symbol (dict of dicts)
    first_key = next(iter(state))
    if isinstance(state[first_key], dict):
        # Multi-symbol mode
        summary_data = []
        total_unrealized_pnl = 0.0
        
        for symbol, data in state.items():
            if symbol == 'params': continue # Skip params if present
            
            price = data.get('price', 0)
            signal = data.get('signal', 'WAIT')
            pos = float(data.get('position', 0))
            pnl = float(data.get('pnl', 0))
            
            total_unrealized_pnl += pnl
            
            summary_data.append({
                "Symbol": symbol,
                "Price": f"${price:.2f}",
                "Signal": signal,
                "Position": pos,
                "PnL": f"${pnl:.2f}"
            })
            
        # Display Total PnL Prominently
        st.metric("💰 Total Unrealized PnL", f"${total_unrealized_pnl:.2f}", delta=f"{total_unrealized_pnl:.2f}")
        
        # Display Summary Table
        st.subheader("📊 Market Overview")
        st.dataframe(pd.DataFrame(summary_data))
        
        # Symbol Selector for Details (Optional, below summary)
        st.markdown("---")
        symbol_list = list(state.keys())
        selected_symbol = st.selectbox("Select Symbol for Details", symbol_list, index=0)
        symbol_data = state.get(selected_symbol, {})
        
    else:
        # Legacy single symbol
        selected_symbol = "BTC/USDT"
        symbol_data = state
        st.metric("💰 Total Unrealized PnL", f"${state.get('pnl', 0):.2f}")

else:
    st.info("Waiting for bot data...")
    symbol_data = {}

# Retrain Button
if st.sidebar.button("🔄 Retrain ML Model"):
    with st.spinner("Training Model..."):
        os.system("python src/train_model.py")
        st.success("Model Retrained!")
        time.sleep(1)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("Strategy Config:")
st.sidebar.json(state.get('params', {}))



# --- PERFORMANCE ---
st.markdown("---")
st.subheader("💰 Performance History")
if os.path.exists('trades.csv'):
    try:
        trades_df = pd.read_csv('trades.csv')
        if not trades_df.empty:
            # Calculate PnL
            pnl = 0.0
            wins = 0
            losses = 0
            open_trades = []
            cumulative_pnl = [0]
            
            for _, row in trades_df.iterrows():
                side = row['side']
                price = row['price']
                qty = row['quantity']
                
                if side in ['buy', 'sell']:
                    open_trades.append({'side': side, 'price': price, 'qty': qty})
                elif 'close' in side:
                    if open_trades:
                        entry = open_trades.pop(0)
                        entry_price = entry['price']
                        exit_price = price
                        trade_qty = min(entry['qty'], qty)
                        
                        trade_pnl = 0
                        if entry['side'] == 'buy':
                            trade_pnl = (exit_price - entry_price) * trade_qty
                        else:
                            trade_pnl = (entry_price - exit_price) * trade_qty
                            
                        pnl += trade_pnl
                        cumulative_pnl.append(pnl)
                        if trade_pnl > 0: wins += 1
                        else: losses += 1

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Realized PnL", f"${pnl:.2f}")
            total_trades = wins + losses
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            m2.metric("Win Rate", f"{win_rate:.1f}%")
            m3.metric("Total Trades", total_trades)
            
            # Equity Curve & Win/Loss Pie
            c_p1, c_p2 = st.columns([2, 1])
            
            with c_p1:
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    y=cumulative_pnl, 
                    mode='lines+markers', 
                    name='Equity',
                    line=dict(color='#00CC96', width=2),
                    fill='tozeroy'
                ))
                fig_equity.update_layout(
                    title="Equity Curve",
                    xaxis_title="Trades",
                    yaxis_title="PnL (USDT)",
                    height=300,
                    template="plotly_dark",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_equity, use_container_width=True)
            
            with c_p2:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Wins', 'Losses'], 
                    values=[wins, losses], 
                    hole=.4,
                    marker=dict(colors=['#00CC96', '#EF553B'])
                )])
                fig_pie.update_layout(
                    title="Win/Loss Ratio",
                    height=300,
                    template="plotly_dark",
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=False
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with st.expander("Trade History"):
                st.dataframe(trades_df.sort_values(by='timestamp', ascending=False))
        else:
            st.info("No trades yet.")
    except Exception as e:
        st.error(f"Error processing trades: {e}")

# --- LOGS ---
st.markdown("---")
st.subheader("📝 System Logs")
log_file = 'bot_clean.log' if os.path.exists('bot_clean.log') else 'bot.log'
if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        st.text_area("Log Output", "".join(lines[-20:]), height=200)

time.sleep(3)
st.rerun()
