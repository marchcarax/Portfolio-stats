We have analyzed 4 basic prediction models, in and out of sample data:
1. ARIMA model
2. Keras LSTM
3. Tensorflow RNN LSTM
4. Montecarlo simulations
5. Gradient boost
6. Stack ARIMA + Keras + Gradient boost

While the models in-sample were very promising, the out-of-sample predictions have been very underwhelming. It's understandable as there are too many factors that can influence a stock price, from fundamental data, random political events, periods of positve or negative sentiment...

According to the Adaptive Markets Hypothesis
(AMH), behavioral biases of market participants, such as
loss aversion, overconfidence, and overreaction, always exist.

Lo AW. The adaptive markets hypothesis. The Journal of Portfolio
Management, 30(5):15–29, 2004.

Does that mean these methods are useless? No. 
We just need to re-interpret (or transform) the data in a way we find better results, without overfitting or creating bias...easy right?

How? what if we create a vector defining periods of up movement, range, down movement? we could create the conditions based on SMA 50 and SMA 200 for example
or what about reinterpret price into code...
or using rolling sharpe ratio...
or using fractals...
or using a highly correlated (better cointegrated) asset and analyze their difference: Asset1 - Asset2 and check their behavior (Pairs trading)...
or maybe transform the data into a mean reverting variable...

There are ways to find utility in these methods. But will they beat buy and hold spy?
more important, would these methods help you in a black swan event?