import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append('..')
from utils.utils import process_multiple_markets
from models.cnn import CNN

sys.path.append('../../database')
sys.path.append('../../database/exchange')
sys.path.append('../../database/interface')
from main import main_exp3 as main
from exchange.enums import MarketFilters, CountryFilters, Databases, MongoURIs, MetaBuilder

class DataPreprocessor:
    """
    Class to preprocess football market data retrieved from a remote database.
    It processes ladders, extracts market data such as prices, LTP, and other key statistics.
    """

    def __init__(self, folder='Soccer/PRO/2023/Jan', uri=MongoURIs.Serverless, 
                 market_filter=MarketFilters.FootballMarketRegex, meta_builder=MetaBuilder.Football, 
                 country_filter=CountryFilters.FootballCountryRegex, database=Databases.Football, 
                 is_multiprocess=False, max_results=50):
        """
        Initialise the DataPreprocessor with parameters for querying the football market data.
        
        Parameters:
        folder (str): The folder path for storing processed data.
        uri (MongoURIs): The MongoDB URI to connect to the database.
        market_filter (MarketFilters): Filter for the specific market.
        meta_builder (MetaBuilder): Metadata builder for the market.
        country_filter (CountryFilters): Filter for countries of interest.
        database (Databases): The database to use.
        is_multiprocess (bool): Whether to use multiprocessing.
        max_results (int): Maximum number of results to retrieve.
        """
        self.folder = folder
        self.uri = uri
        self.market_filter = market_filter
        self.meta_builder = meta_builder
        self.country_filter = country_filter
        self.database = database
        self.is_multiprocess = is_multiprocess
        self.max_results = max_results
        
        # Fetch ladders data using the main function
        self.ladders_list = main(self.folder, self.uri, self.market_filter, self.meta_builder, 
                                 self.country_filter, self.database, self.is_multiprocess, self.max_results)

    def get_prices_and_ltp(self, runner_data):
        """
        Extracts the best back, best lay, and last traded price (LTP) from runner data.
        
        Parameters:
        runner_data (dict): The data dictionary for a runner.
        
        Returns:
        tuple: The best back price, best lay price, and LTP.
        """
        best_back = runner_data.get('atb', [])[0][0] if runner_data.get('atb') else None
        best_lay = runner_data.get('atl', [])[0][0] if runner_data.get('atl') else None
        ltp = runner_data.get('ltp', None)
        return best_back, best_lay, ltp

    def process_single_update(self, update, runner_keys, is_in_play, market_id):
        """
        Process a single update for the market and extract key data points for the runners.
        
        Parameters:
        update (dict): A dictionary containing the market update data.
        runner_keys (list): List of runner IDs.
        is_in_play (bool): Whether the match is in play.
        market_id (str): The unique market ID.
        
        Returns:
        dict: A dictionary of processed data for the market update.
        """
        timestamp = update['pt']

        # Extract best back, best lay, and LTP for each runner
        runner1_back, runner1_lay, runner1_ltp = self.get_prices_and_ltp(update['runners'][runner_keys[0]])
        runner2_back, runner2_lay, runner2_ltp = self.get_prices_and_ltp(update['runners'][runner_keys[1]])
        runner3_back, runner3_lay, runner3_ltp = self.get_prices_and_ltp(update['runners'][runner_keys[2]])

        return {
            'pt': timestamp,
            'id': market_id,
            'runner1_best_back': runner1_back,
            'runner1_best_lay': runner1_lay,
            'runner1_ltp': runner1_ltp,
            'runner2_best_back': runner2_back,
            'runner2_best_lay': runner2_lay,
            'runner2_ltp': runner2_ltp,
            'runner3_best_back': runner3_back,
            'runner3_best_lay': runner3_lay,
            'runner3_ltp': runner3_ltp,
            'inPlay': is_in_play
        }

    def process_ladders(self, ladder_updates):
        """
        Processes multiple ladder updates for a match and extracts relevant market data.
        
        Parameters:
        ladder_updates (list): List of ladder updates.
        
        Returns:
        DataFrame: A DataFrame containing the processed market data.
        """
        processed_data = []
        is_in_play = None
        market_id = ladder_updates[0].get('metadata', None)

        for update in ladder_updates:
            runner_keys = list(update['runners'].keys())
            # Check if the market is in play
            if 'marketDefinition' in update:
                is_in_play = update['marketDefinition'].get('inPlay', None)

            # Process the current update
            processed_update = self.process_single_update(update, runner_keys, is_in_play, market_id)
            processed_data.append(processed_update)
        
        return pd.DataFrame(processed_data)

    def extract_match_data(self):
        """
        Extracts and processes match data from the list of ladders, filtering by in-play status.
        
        Returns:
        list: A list of DataFrames, each containing match data for an individual match.
        """
        match_data_list = []

        for ladders in self.ladders_list:
            # Process ladder data and filter by in-play status
            match_df = self.process_ladders(ladders)
            match_df = match_df[match_df['inPlay'] == True]
            match_data_list.append(match_df)
        
        return match_data_list
