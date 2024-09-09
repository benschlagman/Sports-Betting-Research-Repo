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

### Data Architecture

### Usability
