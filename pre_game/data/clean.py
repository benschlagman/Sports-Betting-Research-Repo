import pandas as pd

class FootyStatsCleaner:
    def __init__(self, original_df):
        """
        Initialises the FootyStatsCleaner class with an original dataframe containing football match data.
        Sets up the columns for the cleaned dataframe that will hold relevant match statistics.

        Parameters:
        original_df (DataFrame): The original DataFrame with raw football match data.
        """
        self.columns = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG',
            'HTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR',
            'AR', 'HP', 'AP', 'B365H', 'B365D', 'B365A', 'PreHXG', 'PreAXG', 'Status'
            ]
        # Initialise an empty DataFrame with the required columns for the cleaned data
        self.new_df = pd.DataFrame(columns=self.columns)
        self.original_df: pd.DataFrame = original_df

    def run(self):
        """
        Main function that runs the cleaning and transformation process on the original data.
        It performs several steps including filtering, sorting, and column transformation.

        Returns:
        DataFrame: The cleaned and processed DataFrame.
        """
        self.filter_complete_matches()
        self.convert_and_sort_by_date()
        self.filter_matches_by_date()
        self.new_df = self.add_columns()
        self.clean()
        return self.new_df

    def clean(self):
        """
        Final cleaning step that removes any rows with missing values or duplicate rows
        from the new dataframe.
        """
        self.new_df = self.new_df.dropna()  # Drop rows with NaN values
        self.new_df = self.new_df.drop_duplicates()  # Remove duplicate rows
    
    def filter_complete_matches(self):
        """
        Filters the original dataframe to include only matches that are marked as 'complete'.
        This ensures that only fully played matches are considered for further processing.
        """
        self.original_df = self.original_df[self.original_df['status'] == 'complete']
    
    def filter_matches_by_date(self, date: str='2023-12-17'):
        """
        Filters matches to include only those played on or before a specified date.

        Parameters:
        date (str): The cutoff date for filtering matches. Matches after this date are excluded.
        """
        self.original_df = self.original_df[self.original_df['date'] <= date]
    
    def convert_and_sort_by_date(self):
        """
        Converts the 'date_unix' column from Unix timestamp to datetime format and sorts
        the dataframe by this date. This ensures chronological ordering of the matches.
        """
        self.original_df['date'] = pd.to_datetime(self.original_df['date_unix'], unit='s')  # Convert Unix time to datetime
        self.original_df = self.original_df.sort_values('date')  # Sort the dataframe by the converted date

    def add_columns(self):
        """
        Adds and transforms relevant columns from the original dataframe to the cleaned dataframe.
        This function maps raw data columns to standardised football statistics columns.

        Returns:
        DataFrame: A DataFrame with the relevant columns filled and match details added.
        """
        # Fill the new dataframe with selected columns from the original dataframe
        self.new_df['Date'] = self.original_df['date']
        self.new_df['HomeTeam'] = self.original_df['home_name']
        self.new_df['AwayTeam'] = self.original_df['away_name']
        self.new_df['FTHG'] = self.original_df['homeGoalCount']
        self.new_df['FTAG'] = self.original_df['awayGoalCount']
        # Determine match result: 'H' for Home win, 'A' for Away win, 'D' for Draw
        self.new_df['FTR'] = self.original_df.apply(lambda row: 'H' if row['homeGoalCount'] > row['awayGoalCount'] else 'A' if row['homeGoalCount'] < row['awayGoalCount'] else 'D', axis=1)
        self.new_df['HTHG'] = self.original_df['ht_goals_team_a']
        self.new_df['HTAG'] = self.original_df['ht_goals_team_b']
        # Determine half-time result
        self.new_df['HTR'] = self.original_df.apply(lambda row: 'H' if row['ht_goals_team_a'] > row['ht_goals_team_b'] else 'A' if row['ht_goals_team_a'] < row['ht_goals_team_b'] else 'D', axis=1)
        # Add other match statistics such as shots, corners, fouls, cards, possession, and betting odds
        self.new_df['HS'] = self.original_df['team_a_shots']
        self.new_df['AS'] = self.original_df['team_b_shots']
        self.new_df['HST'] = self.original_df['team_a_shotsOnTarget']
        self.new_df['AST'] = self.original_df['team_b_shotsOnTarget']
        self.new_df['HC'] = self.original_df['team_a_corners']
        self.new_df['AC'] = self.original_df['team_b_corners']
        self.new_df['HF'] = self.original_df['team_a_fouls']
        self.new_df['AF'] = self.original_df['team_b_fouls']
        self.new_df['HY'] = self.original_df['team_a_yellow_cards']
        self.new_df['AY'] = self.original_df['team_b_yellow_cards']
        self.new_df['HR'] = self.original_df['team_a_red_cards']
        self.new_df['AR'] = self.original_df['team_b_red_cards']
        self.new_df['HP'] = self.original_df['team_a_possession']
        self.new_df['AP'] = self.original_df['team_b_possession']
        self.new_df['B365H'] = self.original_df['odds_ft_1']
        self.new_df['B365D'] = self.original_df['odds_ft_x']
        self.new_df['B365A'] = self.original_df['odds_ft_2']
        self.new_df['PreHXG'] = self.original_df['team_a_xg_prematch']
        self.new_df['PreAXG'] = self.original_df['team_b_xg_prematch']
        self.new_df['Status'] = self.original_df['status']
        self.new_df['RoundID'] = self.original_df['roundID']
        self.new_df['Season'] = self.original_df['season']
        self.new_df['GameWeek'] = self.original_df['game_week']

        return self.new_df
