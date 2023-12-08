"""Main streamlit page"""
# Ignore warnings
import warnings
import datetime

# Data manipulation
import numpy as np
import pandas as pd

# optimizitation
from scipy.optimize import minimize

# Plotting
import plotly.express as px
import streamlit as st

# import quant finance libraries
import yfinance as yf

# import systems

import systems
from systems import sharpe_ratio
import combined_strats as cs


warnings.filterwarnings("ignore")


def main():
    """main"""
    # Begin Streamlit dashboard
    st.set_page_config(layout="wide")

    st.title("Portfolio strategy analysis")
    st.write("by [Marc](https://www.linkedin.com/in/marc-hernandez-fernandez-4481528b/)")

    st.markdown("")

    st.markdown(
        "The goal of the app is to show how following simple investment strategies can outperform the market in the long term. "
        "It follows the naive principal of having an equal-weighted portfolio and each time you add to the portfolio, you are adding equally to all positions. "
    )
    st.markdown("You can also use it to simulate the performance of made-up portfolios.")
    st.markdown(
        "My portfolio consist of a mix of US and European stocks and I try to keep it at less than 20 companies. It changes every 3 to 6 months."
    )
    st.sidebar.caption("Last update: Dec 2023")
    start_date = st.sidebar.date_input("Choose Intial date", datetime.date(2019, 1, 1))

    # Portfolio composition and weight
    # if you have international stocks, remember to put the whole yahoo name with the dot
    portfolio = st.selectbox(
        "Choose portfolio to analyze", ["My portfolio", "CFC-Futures", "M&C", "Other"]
    )
    if portfolio == "My portfolio":
        stocks = [
            "eng.mc",
            "ele.mc",
            "itx.mc",
            "bbva.mc",
            "vid.mc",
            "rep.mc",
            "ibe.mc",
            "or.pa",
            "san.pa",
            "regn",
            "atvi",
            "msft",
            "team",
            "googl",
            "nvda",
            "abnb",
            "sbux"
        ]
    elif portfolio == "CFC-Futures":
        stocks = ["btc-eur","eth-eur","EURUSD=X","GC=F","BZ=F"]
    elif portfolio == "M&C":
        stocks = ["meta", "mc.pa", "googl"]
    else:
        st.markdown(
            "Put stock tickets separated by commas without spaces (e.g. qqq,msft,aapl,ibe.mc)"
        )
        sl = st.text_input("Stock list:")
        stocks = sl.split(",")

    if stocks[0] == "":
        st.write("Waiting stock tickets inputs...")
    else:
        stocks.sort()
        total_stocks = len(stocks)
        weight = [1 / total_stocks] * total_stocks

        # Get data
        @st.cache_data(ttl=86000)
        def get_data(stocks, start_date, end_date):
            return yf.download(stocks, start=start_date, end=end_date)

        df = get_data(stocks, start_date=start_date, end_date="2023-12-31")
        df = df["Adj Close"]
        spy = get_data("spy", start_date=start_date, end_date="2023-12-31")
        spy = spy[["Adj Close"]]
        spy.rename(columns={"Adj Close": "SPY"}, inplace=True)

        st.markdown("#### Simple strategies comparison")

        # call cumulative returns
        returns = cum_returns(df, weight).reset_index()
        returns_spy = (1 + spy.pct_change()[1:]).cumprod().reset_index()
        returns.rename(columns={"Date": "date", 0: "ret"}, inplace=True)

        initial_capital = st.sidebar.slider(
            "Choose initial capital", 10000, 100000, value=50000, step=10000
        )
        add_capital = st.sidebar.slider(
            "Choose amount to periodically add", 1000, 5000, value=1000, step=500
        )
        returns_spy["SPY"] = initial_capital * returns_spy.SPY

        # Strategy 1: Buy 50000 in 2019 and hold
        returns_s1 = returns.copy()
        returns_s1["ret"] = initial_capital * returns_s1.ret

        # Strategy 2: Buy every 3 months
        returns_s2 = returns.copy()
        returns_s2 = systems.compute_strat_2(returns_s2, initial_capital, add_capital, start_date)

        # Strategy 3: Buy everytime RSI dips below 40
        # Define our Lookback period (our sliding window)
        window_length = st.sidebar.slider("Choose window lenght for RSI", 10, 60, value=14, step=1)
        returns_s3 = returns.copy()
        returns_s3 = systems.compute_strat_3(
            returns_s3, initial_capital, add_capital, start_date, window_length
        )

        # Strategy 4: Buy everytime rolling sharpe cycles lower
        # Define our Lookback period (our sliding window)
        returns_s4 = returns.copy()
        returns_s4["sharpe"] = systems.rolling_sharpe(returns_s4.ret.pct_change(), 20, 0)
        buy_signal = st.sidebar.slider(
            "Choose buying line for Rolling sharpe ratio", -10, 0, value=-1, step=1
        )
        sell_signal = st.sidebar.slider(
            "Choose selling line for Rolling sharpe ratio", 0, 10, value=7, step=1
        )

        returns_s4 = systems.compute_strat_4(
            returns_s4, initial_capital, add_capital, start_date, buy_signal, sell_signal
        )

        # Strategy 5: Buy whenever there is low volatily and sell at high volatility periods
        returns_s5 = returns.copy()
        returns_s5 = systems.compute_strat_5(returns_s5,365)

        # Strategy 7: RSI & EMA cross over

        # Strategy 8: Fibonacci levels

        # Strategy 9: Turtle's fast system
        returns_s9 = returns.copy()
        returns_s9 = systems.compute_strat_9(returns_s9, initial_capital, add_capital, 20, 10)

        # mix all signals in a single dataframe
        returns_comb = returns.copy()
        returns_comb["buy_s3"] = returns_s3["buy"].values
        returns_comb["sell_s3"] = returns_s3["sell"].values
        returns_comb["buy_s4"] = returns_s4["buy"].values
        returns_comb["sell_s4"] = returns_s4["sell"].values
        returns_comb["buy_s5"] = returns_s5["buy"].values
        returns_comb["sell_s5"] = returns_s5["sell"].values
        returns_comb["buy_s9"] = returns_s9["buy"].values
        returns_comb["sell_s9"] = returns_s9["sell"].values
        
        # Strategy 10: Mix of signals
        returns_s10 = cs.basic_combination(returns_comb, initial_capital, add_capital, start_date)

        # st.dataframe(returns_s3)

        # Call plotly figures
        df_total = returns_s1.copy()
        returns_spy.rename(columns={"Date": "date", "SPY": "benchmark"}, inplace=True)
        df_total = pd.merge(df_total, returns_spy[["benchmark", "date"]], how="left", on="date")
        df_total["ret_s2"] = returns_s2.ret
        df_total["ret_s3"] = returns_s3.ret
        df_total["ret_s4"] = returns_s4.ret
        #df_total["ret_s5"] = returns_s5.ret
        df_total["ret_s9"] = returns_s9.ret
        df_total["ret_s10"] = returns_s10.ret

        fig = prepare_full_graph(df_total, ["benchmark", "ret", "ret_s2", "ret_s3"])
        st.plotly_chart(fig, use_container_width=True)
        # df_total['dia'] = df_total.date.day
        last_date = df_total.date[-1:].values
        st.write("Last price day is ", pd.Timestamp(last_date[0]).day)
        st.caption("Benchmark is SPY")

        df_ret = returns_s1.set_index("date")
        df_ret["ret_pct"] = df_ret.ret.pct_change()
        df_ret.drop(["ret"], axis=1, inplace=True)
        df_ret = df_ret.resample("MS").sum()
        df_ret.reset_index(inplace=True)
        df_ret["year"] = df_ret["date"].dt.year

        df_ret_spy = returns_spy.set_index("date")
        df_ret_spy["ret_pct"] = df_ret_spy.benchmark.pct_change()
        df_ret_spy.drop(["benchmark"], axis=1, inplace=True)
        df_ret_spy = df_ret_spy.resample("MS").sum()
        df_ret_spy.reset_index(inplace=True)
        df_ret_spy["year"] = df_ret_spy["date"].dt.year

        month = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }

        df_ret["month"] = df_ret["date"].dt.month
        df_ret_spy["month"] = df_ret_spy["date"].dt.month

        df_table = pd.pivot_table(
            df_ret,
            values="ret_pct",
            index=["year"],
            columns=["month"],
            aggfunc=np.sum,
            fill_value=0,
            sort=False,
        )

        df_table_spy = pd.pivot_table(
            df_ret_spy,
            values="ret_pct",
            index=["year"],
            columns=["month"],
            aggfunc=np.sum,
            fill_value=0,
            sort=False,
        )

        df_table.rename(columns=month, inplace=True)
        df_table_spy.rename(columns=month, inplace=True)

        df_table["YTD"] = df_table.sum(axis=1)
        df_table_spy["YTD"] = df_table_spy.sum(axis=1)

        st.write("Table with monthly returns for the portfolio (ret strategy): ")

        st.table(df_table.applymap("{:,.2%}".format))

        with st.expander("Table with monthly returns vs SPY:"):
            df_rest = df_table - df_table_spy
            st.table(df_rest.applymap("{:,.2%}".format))

        # Adding details section so main screen is less convoluted
        risk_free_rate = 0

        with st.expander("See detailed data per strategy"):
            st.markdown("#### Strategy 1: Buy and hold")
            st.markdown("Basic strategy that buys 50K from the period chosen and holds until today")
            mean, stdev = portfolio_info(returns_s1)
            st.write(
                "Portfolio expected annualized return is {} and volatility is {}".format(
                    mean, stdev
                )
            )
            st.write("Portfolio sharpe ratio is {0:0.2f}".format(sharpe_ratio(returns_s1.ret.pct_change(), risk_free_rate)))

            st.markdown("#### Strategy 2: Buy & sells periodically")
            st.markdown("After an initial capital investment, we add capital every 3 months and sell every 5 months a 2% of the portfolio")
            mean, stdev = portfolio_info(returns_s2)
            st.write(
                "Portfolio expected annualized return is {} and volatility is {}".format(
                    mean, stdev
                )
            )
            st.write("Portfolio sharpe ratio is {0:0.2f}".format(sharpe_ratio(returns_s2.ret.pct_change(), risk_free_rate)))

            st.markdown("#### Strategy 3: Buy after every month when RSI < 30 and sells when > 80")
            st.markdown(
                "After an initial capital investment, we add capital every month when RSI is lower than 30 and sells when above 80"
            )
            mean, stdev = portfolio_info(returns_s3.drop(["rsi", "buy", "sell"], axis=1))
            st.write(
                "Portfolio expected annualized return is {} and volatility is {}".format(
                    mean, stdev
                )
            )
            st.write("Portfolio sharpe ratio is {0:0.2f}".format(sharpe_ratio(returns_s3.ret.pct_change(), risk_free_rate)))

            st.markdown("##### RSI graph")
            fig = px.line(returns_s3, x="date", y="rsi")
            fig.add_hline(y=30, line_color="green", line_dash="dash")
            fig.add_hline(y=80, line_color="red", line_dash="dash")
            st.plotly_chart(fig, use_container_width=False)
            st.write("Last RSI data point is {}".format(returns_s3.rsi[-1:].values))

            st.markdown("##### Buy signals for Strat 3")
            fig = px.line(returns_s3, x="date", y=["buy", "sell"])
            st.plotly_chart(fig, use_container_width=False)


        st.markdown("#### Advanced strategies comparison")

        fig = prepare_full_graph(df_total, ["ret", "ret_s4", "ret_s10"])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Benchmark is 50/50 portfolio")

        df_ret = returns_s10.set_index("date")
        df_ret["ret_pct"] = df_ret.ret.pct_change()
        df_ret.drop(["ret"], axis=1, inplace=True)
        df_ret = df_ret.resample("MS").sum()
        df_ret.reset_index(inplace=True)
        df_ret["year"] = df_ret["date"].dt.year

        month = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }

        df_ret["month"] = df_ret["date"].dt.month

        df_table = pd.pivot_table(
            df_ret,
            values="ret_pct",
            index=["year"],
            columns=["month"],
            aggfunc=np.sum,
            fill_value=0,
            sort=False,
        )

        df_table.rename(columns=month, inplace=True)

        df_table["YTD"] = df_table.sum(axis=1)

        st.write("Table with monthly returns for strategy 10: ")

        st.table(df_table.applymap("{:,.2%}".format))

        with st.expander("See detailed data per strategy"):
            st.markdown("#### Strategy 4: Buy everytime rolling sharpe cycles lower")
            st.markdown(
                "After an initial capital investment, we add capital every month when rolling Sharpe ratio cycles lower than 0 and we take capital every 3 months when sharpe ratio higher than 0.6"
            )
            mean, stdev = portfolio_info(returns_s4.drop(["sharpe", "buy", "sell"], axis=1))
            st.write(
                "Portfolio expected annualized return is {} and volatility is {}".format(
                    mean, stdev
                )
            )
            st.write("Portfolio sharpe ratio is {0:0.2f}".format(sharpe_ratio(returns_s4.ret.pct_change(), risk_free_rate=risk_free_rate)))

            st.markdown("##### Rolling sharpe graph")
            fig = px.line(returns_s4, x="date", y="sharpe")
            fig.add_hline(y=buy_signal, line_color="green", line_dash="dash")
            fig.add_hline(y=sell_signal, line_color="red", line_dash="dash")
            st.plotly_chart(fig, use_container_width=False)
            st.write("Last rolling sharpe data point is {}".format(returns_s4.sharpe[-1:].values))

            st.markdown("##### Buy & Sell signals for Strat 4")
            fig = px.line(returns_s4, x="date", y=["buy", "sell"])
            st.plotly_chart(fig, use_container_width=False)

            st.markdown("#### Strategy 10: Basic combination of systems")
            st.markdown("Basic combination from strategy 3, strategy 4 & strategy 9")
            mean, stdev = portfolio_info(returns_s10[["date", "ret"]])
            st.write(
                "Portfolio expected annualized return is {} and volatility is {}".format(
                    mean, stdev
                )
            )
            st.write("Portfolio sharpe ratio is {0:0.2f}".format(sharpe_ratio(returns_s10.ret.pct_change(), risk_free_rate)))

            st.markdown("##### Buy&Sell signals for Strat 10")
            fig = px.line(returns_s10, x="date", y=["buy", "sell"])
            st.plotly_chart(fig, use_container_width=False)


        st.markdown("#### What to buy")
        st.markdown("")
        st.markdown(
            "We will use Efficient Frontier to find the optimal weight allocation of the Portfolio that returns the best sharpe ratio. "
            "We will then print the top 3 stocks and their weights to gives us an idea where we could potentially add to the portfolio (if current weight does not exceed optimal weight). "
        )
        if len(stocks) > 2:
            allocation = efficient_frontier(df, stocks)
            df_ef = pd.DataFrame.from_dict(allocation, orient="index", columns=["weights"])
            st.write(df_ef.sort_values("weights", ascending=False)[:3].index.tolist())
        else:
            st.write("No enought stocks to create optimal portfolio.")

        st.markdown("")
        st.markdown(
            "Another way to find the optimal allocation is using the optimizer function from scipy. We will use it to find the weights that mazimize the sharpe ratio "
        )
        if len(stocks) > 2:
            allocation = optimize_weights(df.pct_change(), 4, stocks)
            st.write(allocation[:3].index.tolist())
        else:
            st.write("No enought stocks to create optimal portfolio.")


def prepare_full_graph(df: pd.DataFrame, list_y: list):
    """prepares line graph"""
    return px.line(
        df,
        x="date",
        y=list_y,
        color_discrete_sequence=px.colors.qualitative.G10,
        render_mode="SVG",
    )


def portfolio_info(stocks: pd.DataFrame):
    """Function that calculates portfolio returns and volatility"""
    stocks.drop(["date"], axis=1, inplace=True)
    mean_daily_returns = stocks.pct_change().mean()
    cov_data = stocks.pct_change().cov()
    portfolio_return = round(np.sum(mean_daily_returns) * 252, 2)
    # calculate annualized portfolio volatility
    portfolio_std_dev = round(np.sqrt(cov_data) * np.sqrt(252), 2)

    return portfolio_return * 100, float(portfolio_std_dev.values) * 100



def cum_returns(stocks: pd.DataFrame, wts: list):
    """Returns cumulative returns of the portfolio applying the assigned weights"""
    weighted_returns = wts * stocks.pct_change()[1:]
    weighted_returns = pd.DataFrame(weighted_returns)
    port_ret = weighted_returns.sum(axis=1)
    return (port_ret + 1).cumprod()


@st.cache_data(ttl=604800)
def efficient_frontier(df, stocks, num_runs=100):
    """function that calculates efficient frontier for portfolio optimization"""
    log_ret = np.log(df / df.shift(1))

    all_weights = np.zeros((num_runs, len(stocks)))
    ret_arr = np.zeros(num_runs)
    vol_arr = np.zeros(num_runs)
    sharpe_arr = np.zeros(num_runs)

    for ind in range(num_runs):
        # Create Random Weights
        weights = np.array(np.random.random(len(stocks)))

        # Rebalance Weights
        weights = weights / np.sum(weights)

        # Save Weights
        all_weights[ind, :] = weights

        # Expected Return
        ret_arr[ind] = np.sum((log_ret.mean() * weights) * 252)

        # Expected Variance
        vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

        # Sharpe Ratio
        sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

    allocation = [i * 100 for i in all_weights[sharpe_arr.argmax(), :]]
    stock_dict = dict(zip(stocks, allocation))

    return stock_dict


@st.cache_data(ttl=604800)
def optimize_weights(returns: pd.DataFrame, risk_free_rate: float, stock_list: list):
    """finds the weights that maximize the sharpe ratio"""
    n = returns.shape[1]
    initial_weights = np.ones(n) / n
    bounds = [(0, 1) for i in range(n)]
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    def neg_sharpe_ratio(weights: np.array, returns: pd.DataFrame, risk_free_rate: float):
        """Function that returns the negative of the sharpe ratio"""
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_r = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_r

    result = minimize(
        fun=neg_sharpe_ratio,
        x0=initial_weights,
        args=(returns, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    optimized_weights = result.x
    res = pd.DataFrame(data=optimized_weights, index=stock_list, columns=["res"]).sort_values(
        by="res", ascending=False
    )
    return res


if __name__ == "__main__":
    main()
