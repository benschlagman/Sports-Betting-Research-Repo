# Sports-Betting-Research-Repo

## Overview
This project investigates the application of deep learning to sports betting, specifically targeting online exchanges like Betfair. The work was conducted in collaboration with Quant Sports Trading (QST) and is structured around three core experiments. The project integrates various types of sports data — pre-game, in-play, and exchange data — to facilitate sophisticated model building and in-depth analysis in sports betting.

The project’s research is organised into three experiments, detailed in Chapters 3, 4, and 5 of the thesis:

- **Chapter 3 (Data Exploration and Infrastructure)**: Focuses on creating the research platform and database architecture, which is implemented in the `database` folder.
- **Chapter 4 (Baseline Models for Pre-Game Football Predictions)**: Develops a series of baseline value betting models to classify the outcome of upcoming football matches based on historical pre-game data. The code for this is located in the `pre_game` folder.
- **Chapter 5 (Deep Learning for In-Play Momentum Betting)**: Implements deep learning models for in-play momentum betting. The corresponding code is in the `in_play` folder.

These experiments collectively enable advanced model training, backtesting, and real-time prediction for sports betting strategies.

## Setup

Clone the repository and run `pip install -r requirements.txt` to install dependencies. Run the following commands: `pip install pipreqs` and `pipreqs --force` to generate the `requirements.txt` file and update dependencies during development. To run the MongoDB and AWS pipeline, run `database/main.py`. To run the models for pre-game predictions, execute the notebook located at `pre_game/notebooks/model.ipynb`. For in-play models, navigate to `in_play/notebooks` and run each of the four models, as each one has its own dedicated notebook.

## Database

### Data

The platform combines data from multiple sources to provide comprehensive insights for predictive modelling and live sports trading:

- **Pre-game Data**: Offers a historical view of match data and key statistics, which is essential for building predictive models before games start. This data is sourced from providers like FootyStats.
- **In-Play Data**: Provides a live state-of-game data, capturing details such as player positions, passes, and shots. This data is crucial for assessing player performance and dynamically updating betting odds during live events. It is gathered from providers like StatsBomb.
- **Exchange Data**: Sourced from Betfair, this data provides detailed information on betting market trends, including back and lay prices, traded volumes, and order book updates. This data is essential for financial analysis and real-time decision-making in betting markets.

These diverse data types are crucial for enabling complex analysis beyond what basic models would allow, offering the ability to integrate real-time events with historical patterns to enhance betting strategies.

### Data Architecture and Abstraction

To facilitate efficient data querying, integration, and research, the platform has been designed with a robust architecture:

- **Abstraction Layers**: Each data source (pre-game, in-play, and exchange) is abstracted with its own layer, allowing for easy access and querying. This abstraction ensures that researchers can seamlessly interact with the data without needing to worry about the underlying complexities.
- **Pre-processing**: Data is pre-processed to match the real-life scenarios of live trading data. This includes normalising, cleaning, and aligning data from different sources to ensure it can be used in real-time models and backtesting environments.

The data platform leverages a combination of **MongoDB** (NoSQL) and **SQL databases**:

- **MongoDB** is used for handling the high-frequency and unstructured in-play and exchange data. This NoSQL database allows for the storage and rapid retrieval of real-time, complex datasets, such as millisecond-level updates from Betfair.
- **SQL databases** are employed for more structured historical data, such as pre-game statistics and player records. This ensures efficient querying and storage for large volumes of pre-game data used in predictive models.

By integrating these databases, the platform supports robust data retrieval and analysis, significantly reducing the time and effort needed for backtesting and iterating sports betting strategies.

## Pre-game

Experiment 2 consists of a complete machine learning pipeline which utilises the `data`, `features`, and `pipeline` packages. The goal of Experiment 2 was to enhance the predictive accuracy of machine learning models by leveraging pre-game data to classify the outcomes of football matches before they begin. This serves as a benchmark for more advanced, in-play models.

### Key Aspects of Data Preparation Include:

- **Feature Engineering**: Key statistics from pre-game data are transformed into features that are fed into machine learning models. These include metrics such as team performance, player stats, historical match results, and expected goals (xG).
  
- **Data Cleaning and Pre-processing**: Pre-game data is normalised and cleaned to ensure consistent quality and reliability. This involves handling missing values, removing irrelevant data points, and formatting the data into a usable structure for the models.
  
- **Training, Validation, and Testing Split**: The data is split into training, validation, and testing sets to ensure that the models generalise well to unseen data. Historical match data from previous seasons is used to train and evaluate the models' performance.
  
- **Model Selection**: A variety of machine learning models, such as XGBoost, Support Vector Machines (SVM), and Random Forest Classifiers, are implemented to predict match outcomes like home wins, away wins, or draws. These models are tuned using hyperparameter optimisation techniques to achieve the best possible results.

By applying these steps, the models in Experiment 2 establish a robust foundation for predicting football match outcomes before they start, providing essential benchmarks for more complex in-play predictions.


## In-play


