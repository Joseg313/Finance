# Stock Market Visualizer
# A Streamlit web application for financial market analysis.

# Import necessary libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import base64
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Market Visualizer",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    """Fetches historical stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check the ticker symbol.")
            return None
        return data
    except Exception as e:
        st.error(f"An error occurred while fetching data for {ticker}: {e}")
        return None

def calculate_moving_average(data, window, ma_type='SMA'):
    """Calculates Simple or Exponential Moving Average."""
    if ma_type == 'SMA':
        return data['Close'].rolling(window=window).mean()
    elif ma_type == 'EMA':
        return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_financial_ratios(ticker):
    """Fetches key financial ratios and information for a stock."""
    try:
        stock_info = yf.Ticker(ticker).info
        ratios = {
            "Market Cap": stock_info.get("marketCap"),
            "Forward P/E": stock_info.get("forwardPE"),
            "Trailing P/E": stock_info.get("trailingPE"),
            "Price to Sales (TTM)": stock_info.get("priceToSalesTrailing12Months"),
            "Price to Book": stock_info.get("priceToBook"),
            "Enterprise Value to EBITDA": stock_info.get("enterpriseToEbitda"),
            "Profit Margin": stock_info.get("profitMargins"),
            "Operating Margin (TTM)": stock_info.get("operatingMargins"),
            "Return on Assets (TTM)": stock_info.get("returnOnAssets"),
            "Return on Equity (TTM)": stock_info.get("returnOnEquity"),
            "Revenue Growth (YoY)": stock_info.get("revenueGrowth"),
            "Earnings Growth (YoY)": stock_info.get("earningsGrowth"),
            "Beta": stock_info.get("beta"),
            "52 Week High": stock_info.get("fiftyTwoWeekHigh"),
            "52 Week Low": stock_info.get("fiftyTwoWeekLow"),
            "Dividend Yield": stock_info.get("dividendYield"),
        }
        return {k: v for k, v in ratios.items() if v is not None}
    except Exception as e:
        st.warning(f"Could not fetch financial ratios for {ticker}: {e}")
        return {}

def create_download_link(fig, filename, filetype, link_text):
    """Creates a download link for a Plotly figure."""
    if filetype == 'html':
        buffer = BytesIO()
        fig.write_html(buffer)
        b64 = base64.b64encode(buffer.getvalue()).decode()
    elif filetype == 'png':
        buffer = BytesIO(fig.to_image(format="png"))
        b64 = base64.b64encode(buffer.read()).decode()
    
    href = f'<a href="data:file/{filetype};base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# --- Sidebar ---
st.sidebar.title("📈 Stock Market Visualizer")
st.sidebar.markdown("---")

# Session state initialization
if 'tickers' not in st.session_state:
    st.session_state.tickers = ['AAPL', 'GOOGL']

# Configuration upload
st.sidebar.header("Configuration")
uploaded_config = st.sidebar.file_uploader("Upload Configuration File", type=['json'])
if uploaded_config is not None:
    try:
        config = json.load(uploaded_config)
        st.session_state.tickers = config.get('tickers', ['AAPL', 'GOOGL'])
        # You can extend this to load other settings as well
        st.sidebar.success("Configuration loaded successfully!")
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON file. Please upload a valid configuration file.")

# Main stock selection
st.sidebar.header("Stock Selection")
main_ticker = st.sidebar.text_input("Enter a Stock Ticker (e.g., AAPL)", value=st.session_state.tickers[0])
tickers_for_comparison = st.sidebar.multiselect(
    "Compare with other stocks:",
    options=['SPY', 'QQQ', 'DIA', 'TSLA', 'MSFT', 'AMZN', 'NVDA', 'META'],
    default=[t for t in st.session_state.tickers if t != main_ticker]
)

all_tickers = [main_ticker] + tickers_for_comparison
all_tickers = sorted(list(set(filter(None, all_tickers)))) # Remove duplicates and empty strings

# Date range selection
st.sidebar.header("Date Range")
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", pd.to_datetime('2023-01-01'))
end_date = col2.date_input("End Date", pd.to_datetime('today'))

st.sidebar.markdown("---")

# --- Main Application ---
st.title(f"Analysis for: {main_ticker}")

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Chart", " ratios", "📈 Portfolio Tracker", "🔗 Correlation Analysis"])

# --- Tab 1: Main Chart and Technical Analysis ---
with tab1:
    st.header("Interactive Candlestick Chart")
    
    # Fetch data for the main ticker
    data = get_stock_data(main_ticker, start_date, end_date)

    if data is not None:
        # Technical Indicators Configuration
        st.subheader("Technical Indicators")
        indicator_expander = st.expander("Configure Indicators", expanded=True)
        with indicator_expander:
            c1, c2, c3 = st.columns(3)
            # Moving Averages
            with c1:
                st.markdown("**Moving Averages**")
                show_ma = st.checkbox("Show Moving Averages", value=True)
                ma1_period = st.number_input("MA 1 Period", min_value=5, max_value=200, value=20, step=1)
                ma2_period = st.number_input("MA 2 Period", min_value=5, max_value=200, value=50, step=1)
                ma_type = st.selectbox("MA Type", ['SMA', 'EMA'])

            # Bollinger Bands
            with c2:
                st.markdown("**Bollinger Bands**")
                show_bb = st.checkbox("Show Bollinger Bands", value=True)
                bb_period = st.number_input("BB Period", min_value=5, max_value=200, value=20, step=1)
                bb_std_dev = st.number_input("BB Std. Dev.", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

            # RSI
            with c3:
                st.markdown("**Relative Strength Index (RSI)**")
                show_rsi = st.checkbox("Show RSI", value=True)
                rsi_period = st.number_input("RSI Period", min_value=5, max_value=50, value=14, step=1)
                rsi_overbought = st.number_input("Overbought Threshold", min_value=50, max_value=90, value=70, step=1)
                rsi_oversold = st.number_input("Oversold Threshold", min_value=10, max_value=50, value=30, step=1)

        # Calculate indicators
        if show_ma:
            data[f'{ma_type} {ma1_period}'] = calculate_moving_average(data, ma1_period, ma_type)
            data[f'{ma_type} {ma2_period}'] = calculate_moving_average(data, ma2_period, ma_type)
        if show_bb:
            data['BB_Upper'], data['BB_Lower'] = calculate_bollinger_bands(data, bb_period, bb_std_dev)
        if show_rsi:
            data['RSI'] = calculate_rsi(data, rsi_period)

        # Create plot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.1, row_heights=[0.7, 0.3])

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candlestick'),
                      row=1, col=1)

        # Add Moving Averages
        if show_ma:
            fig.add_trace(go.Scatter(x=data.index, y=data[f'{ma_type} {ma1_period}'], mode='lines', name=f'{ma_type} {ma1_period}', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data[f'{ma_type} {ma2_period}'], mode='lines', name=f'{ma_type} {ma2_period}', line=dict(color='purple')), row=1, col=1)

        # Add Bollinger Bands
        if show_bb:
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='Upper Band', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='Lower Band', line=dict(color='gray', dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

        # Add RSI plot
        if show_rsi:
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='cyan')), row=2, col=1)
            fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", annotation_text="Overbought", row=2, col=1)
            fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", annotation_text="Oversold", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)

        # Update layout
        fig.update_layout(
            title=f'{main_ticker} Stock Price',
            yaxis_title='Price (USD)',
            xaxis_rangeslider_visible=False,
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showspikes=True)
        fig.update_yaxes(showspikes=True)
        
        st.plotly_chart(fig, use_container_width=True)

        # Chart Export
        st.markdown("---")
        st.subheader("Export Chart")
        c1_export, c2_export = st.columns(2)
        with c1_export:
            st.markdown(create_download_link(fig, f"{main_ticker}_chart.html", "html", "Download as HTML"), unsafe_allow_html=True)
        with c2_export:
            st.markdown(create_download_link(fig, f"{main_ticker}_chart.png", "png", "Download as PNG"), unsafe_allow_html=True)

# --- Tab 2: Financial Ratios ---
with tab2:
    st.header(f"Key Financial Ratios for {main_ticker}")
    ratios = get_financial_ratios(main_ticker)
    if ratios:
        cols = st.columns(4)
        i = 0
        for name, value in ratios.items():
            col = cols[i % 4]
            if isinstance(value, float):
                formatted_value = f"{value:,.2f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = value
            col.metric(label=name, value=formatted_value)
            i += 1
    else:
        st.info("No financial ratios available for this ticker.")

# --- Tab 3: Portfolio Tracker ---
with tab3:
    st.header("Portfolio Tracker")
    st.markdown("Upload a CSV or Excel file with 'Ticker' and 'Shares' columns.")
    
    # Download template
    portfolio_template = pd.DataFrame({'Ticker': ['AAPL', 'GOOG'], 'Shares': [10, 5]})
    csv = portfolio_template.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Portfolio Template (CSV)",
        data=csv,
        file_name='portfolio_template.csv',
        mime='text/csv',
    )
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                portfolio_df = pd.read_csv(uploaded_file)
            else:
                portfolio_df = pd.read_excel(uploaded_file)

            if 'Ticker' in portfolio_df.columns and 'Shares' in portfolio_df.columns:
                portfolio_tickers = portfolio_df['Ticker'].tolist()
                
                @st.cache_data
                def get_latest_prices(tickers):
                    prices = {}
                    for ticker in tickers:
                        try:
                            # Fetch last day's data to get the most recent close price
                            data = yf.download(ticker, period="1d", progress=False)
                            if not data.empty:
                                prices[ticker] = data['Close'].iloc[-1]
                        except Exception:
                            prices[ticker] = np.nan
                    return prices

                latest_prices = get_latest_prices(tuple(portfolio_tickers))
                
                portfolio_df['Current Price'] = portfolio_df['Ticker'].map(latest_prices)
                portfolio_df.dropna(subset=['Current Price'], inplace=True)
                portfolio_df['Current Value'] = portfolio_df['Shares'] * portfolio_df['Current Price']
                
                total_portfolio_value = portfolio_df['Current Value'].sum()
                
                st.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")
                
                st.subheader("Portfolio Allocation")
                fig_pie = go.Figure(data=[go.Pie(labels=portfolio_df['Ticker'], values=portfolio_df['Current Value'], hole=.3)])
                fig_pie.update_layout(title_text='Portfolio Allocation by Value')
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.subheader("Detailed View")
                st.dataframe(portfolio_df.style.format({
                    'Shares': '{:,.2f}',
                    'Current Price': '${:,.2f}',
                    'Current Value': '${:,.2f}'
                }))

            else:
                st.error("The uploaded file must contain 'Ticker' and 'Shares' columns.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# --- Tab 4: Correlation Analysis ---
with tab4:
    st.header("Stock Correlation Analysis")
    if len(all_tickers) > 1:
        # Fetch adjusted close prices for all selected stocks
        @st.cache_data
        def get_multiple_adj_close(tickers, start, end):
            adj_close_df = pd.DataFrame()
            for ticker in tickers:
                data = get_stock_data(ticker, start, end)
                if data is not None:
                    adj_close_df[ticker] = data['Adj Close']
            return adj_close_df.dropna()

        adj_close_data = get_multiple_adj_close(tuple(all_tickers), start_date, end_date)
        
        if not adj_close_data.empty:
            returns = adj_close_data.pct_change().dropna()
            correlation_matrix = returns.corr()

            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='Viridis',
                zmin=-1, zmax=1
            ))
            fig_corr.update_layout(
                title='Correlation Matrix of Stock Returns',
                xaxis_nticks=len(correlation_matrix.columns),
                yaxis_nticks=len(correlation_matrix.columns)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.subheader("Correlation Matrix Data")
            st.dataframe(correlation_matrix.style.background_gradient(cmap='viridis').format("{:.2f}"))
        else:
            st.warning("Could not fetch sufficient data for correlation analysis.")
    else:
        st.info("Please select at least two stocks in the sidebar for correlation analysis.")


# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.header("Save Configuration")
config_to_save = {'tickers': all_tickers}
config_json = json.dumps(config_to_save, indent=4)
st.sidebar.download_button(
    label="Download Configuration",
    data=config_json,
    file_name='stock_visualizer_config.json',
    mime='application/json',
)
st.sidebar.info("Save your current stock selections to a file to load them later.")
