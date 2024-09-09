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
from main import main  
from exchange.enums import MarketFilters, CountryFilters, Databases, MongoURIs, MetaBuilder

class DataPreprocessor:
    def __init__(self, folder='Soccer/PRO/2023/Jan', uri=MongoURIs.Serverless, 
                 market_filter=MarketFilters.FootballMarketRegex, meta_builder=MetaBuilder.Football, 
                 country_filter=CountryFilters.FootballCountryRegex, database=Databases.Football, 
                 is_multiprocess=False, max_results=50):
        
        self.folder = folder
        self.uri = uri
        self.market_filter = market_filter
        self.meta_builder = meta_builder
        self.country_filter = country_filter
        self.database = database
        self.is_multiprocess = is_multiprocess
        self.max_results = max_results
        
        # Call the main function to get ladders
        self.ladders_list = main(self.folder, self.uri, self.market_filter, self.meta_builder, 
                                 self.country_filter, self.database, self.is_multiprocess, self.max_results)

    def get_prices_and_ltp(self, runner_data):
        best_back = runner_data.get('atb', [])[0][0] if runner_data.get('atb') else None
        best_lay = runner_data.get('atl', [])[0][0] if runner_data.get('atl') else None
        ltp = runner_data.get('ltp', None)
        return best_back, best_lay, ltp

    def process_single_update(self, update, runner_keys, current_inPlay, id):
        pt = update['pt']

        runner1_best_back, runner1_best_lay, runner1_ltp = self.get_prices_and_ltp(update['runners'][runner_keys[0]])
        runner2_best_back, runner2_best_lay, runner2_ltp = self.get_prices_and_ltp(update['runners'][runner_keys[1]])
        runner3_best_back, runner3_best_lay, runner3_ltp = self.get_prices_and_ltp(update['runners'][runner_keys[2]])

        return {
            'pt': pt,
            'id': id,
            'runner1_best_back': runner1_best_back,
            'runner1_best_lay': runner1_best_lay,
            'runner1_ltp': runner1_ltp,
            'runner2_best_back': runner2_best_back,
            'runner2_best_lay': runner2_best_lay,
            'runner2_ltp': runner2_ltp,
            'runner3_best_back': runner3_best_back,
            'runner3_best_lay': runner3_best_lay,
            'runner3_ltp': runner3_ltp,
            'inPlay': current_inPlay
        }

    def process_ladders(self, ladders):
        extracted_data = []
        current_inPlay = None
        id = ladders[0].get('metadata', None)

        for update in ladders:
            runner_keys = list(update['runners'].keys())
            if 'marketDefinition' in update:
                current_inPlay = update['marketDefinition'].get('inPlay', None)

            processed_data = self.process_single_update(update, runner_keys, current_inPlay, id)
            extracted_data.append(processed_data)
        
        return pd.DataFrame(extracted_data)

    def extract_match_data(self):
        df_per_match = []

        for ladders in self.ladders_list:
            df_match = self.process_ladders(ladders)
            df_match = df_match[df_match['inPlay'] == True]
            df_per_match.append(df_match)
        
        return df_per_match
