# Stock Prediction App

This web app, based on Streamlit, has 7 pages designed to provide comprehensive stock analysis and prediction capabilities. The primary focus is the **"Prediction" page**, where you can leverage two different LSTM models to forecast stock prices. These models were created in Jupyter Notebook files included in this repository. You have the flexibility to modify these files to develop new versions of the models or use the pre-trained `.h5` files that already contain the models.

## Pages

1. **Login**
   - Secure access to the app, ensuring user data protection and personalized experiences.

2. **Hello**
   - A welcome page that introduces users to the app and provides an overview of its functionalities.

3. **Info**
   - Displays detailed information about selected stocks, including current prices, historical data, and key financial metrics.

4. **Prediction**
   - The core of the app, utilizing two distinct LSTM models to predict future stock prices. Users can input specific stock symbols and parameters to generate predictions and visualize the results.
   
5. **Compare**
   - Allows users to compare the performance and predictions of different stocks side by side, facilitating better investment decisions.

6. **News**
   - Fetches the latest news related to selected stocks, providing insights into market trends and factors influencing stock prices.

7. **Twitter**
   - Analyzes recent tweets about selected stocks to gauge market sentiment and identify potential impacts on stock prices.

## How to Use

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/stock-prediction-app.git
   cd stock-prediction-app

## Install Dependencies
Ensure you have Python installed, then run:

  pip install -r requirements.txt

## Run the App

  streamlit run app.py

## Model Files

  The .h5 files in the repository contain pre-trained LSTM models. You can use these directly or modify the Jupyter Notebooks to train new models.
  
## Dependencies
  Streamlit: For building the web interface.
  TensorFlow/Keras: For creating and running LSTM models.
  yfinance: To fetch and process stock data.
  Pandas: For data manipulation and analysis.
  Plotly: For data visualization.
  Other libraries: As listed in requirements.txt.
  
## Modifying the Models
  The Jupyter Notebook files provided allow you to understand and modify the LSTM models used for stock prediction. You can experiment with different architectures, hyperparameters, and datasets to improve prediction accuracy. Once satisfied, save your model as an .h5 file and update the app to use your new model.

## Contributing
  Feel free to contribute to this project by opening issues or submitting pull requests. Your contributions are highly appreciated!

## License
  This project is licensed under the MIT License. See the LICENSE file for details.
