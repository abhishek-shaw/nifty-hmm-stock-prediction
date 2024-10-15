# nifty-hmm-stock-prediction
Predicting Nifty trends using Hidden Markov Model Machine learning to detect future trends

This program implements a Hidden Markov Model to predict stock market trends using the nifty50.csv file. Here's a breakdown of what the code does:

- We import the necessary libraries: pandas for data manipulation, numpy for numerical operations, hmmlearn for the Hidden Markov Model, matplotlib for plotting, and sklearn for data preprocessing.
- We load the data from the CSV file, preprocess it by calculating returns, and scale the data.
- We define and train a Hidden Markov Model with 5 states, representing our 5 trend categories (Strong Bullish, Mild Bullish, Sideways, Mild Bearish, Strong Bearish).
- We use the trained model to predict hidden states for our data and map these states to trend labels.
- We calculate support and resistance levels using a simple rolling window approach.
- We plot the results, showing the close price with support and resistance levels, as well as the predicted market trends.
- Finally, we predict the future trend and the next day's price based on the last available data point.

To run this program, you'll need to install the required libraries. You can do this by running:
```
pip install pandas numpy matplotlib mplfinance scikit-learn hmmlearn

# Run the program

python hmm.py
```
Make sure the 'nifty50.csv' file is in the same directory as your Python script.

View [chart.html](https://abhishek-shaw.github.io/nifty-hmm-stock-prediction/chart.html) for Predection

### Disclaimer :
This implementation provides a basic framework for predicting stock market trends using HMM. However, please note that predicting stock market movements is a complex task, and this simple model may not capture all the intricacies of real market behavior. It's always recommended to use multiple analysis techniques and consult with financial experts before making investment decisions.
