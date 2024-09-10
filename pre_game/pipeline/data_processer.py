import pandas as pd
from features.individual_stats import IndividualTeamStats
from utils.model_utils import Feature
import json


class XTableConstructor:
    """
    A class to construct a feature table (X) for machine learning models from football match statistics.
    Allows for selecting specific features to be included in the table.
    """
    def __init__(self, **kwargs):
        # Initialise boolean flags for each feature type based on keyword arguments
        self.include_goal_stats: bool = kwargs.get(Feature.GOAL_STATS.value, False)
        self.include_shooting_stats: bool = kwargs.get(Feature.SHOOTING_STATS.value, False)
        self.include_possession_stats: bool = kwargs.get(Feature.POSSESSION_STATS.value, False)
        self.include_result: bool = kwargs.get(Feature.RESULT.value, False)
        self.include_odds: bool = kwargs.get(Feature.ODDS.value, False)
        self.include_xg: bool = kwargs.get(Feature.XG.value, False)
        self.include_home_away_results: bool = kwargs.get(Feature.HOME_AWAY_RESULTS.value, False)
        self.include_conceded_stats: bool = kwargs.get(Feature.CONCEDED_STATS.value, False)
        self.include_last_n_matches: bool = kwargs.get(Feature.LAST_N_MATCHES.value, False)
        self.include_win_streak: bool = kwargs.get(Feature.WIN_STREAK.value, False)

    def construct_row(self, match_row, home_team, away_team, home_team_stats, away_team_stats) -> dict:
        """
        Constructs a single row of feature values for a given match, based on the specified team stats.

        Parameters:
        match_row (Series): Original row from the match data.
        home_team (str): Name of the home team.
        away_team (str): Name of the away team.
        home_team_stats (DataFrame): Statistics for the home team.
        away_team_stats (DataFrame): Statistics for the away team.

        Returns:
        dict: A dictionary representing the feature row for the match.
        """
        row = {
            'HT': home_team,
            'AT': away_team,
        }

        if self.include_result:
            row = self.add_result_percentage(row, home_team_stats, away_team_stats)

        if self.include_possession_stats:
            row = self.add_possession_stats(row, home_team_stats, away_team_stats)

        if self.include_home_away_results:
            row = self.add_home_away_result_percentage(row, home_team_stats, away_team_stats)

        if self.include_shooting_stats:
            row = self.add_shooting_stats(row, home_team_stats, away_team_stats)

        if self.include_goal_stats:
            row = self.add_goal_stats(row, home_team_stats, away_team_stats)

        if self.include_conceded_stats:
            row = self.add_conceded_stats(row, home_team_stats, away_team_stats)

        if self.include_last_n_matches:
            row = self.add_last_n_matches_stats(row, home_team_stats, away_team_stats)

        if self.include_win_streak:
            row = self.add_win_streak(row, home_team_stats, away_team_stats)

        if self.include_odds:
            row = self.add_odds(row, match_row)

        if self.include_xg:
            row = self.add_xg(row, match_row)

        return row

    def divide(self, stat_dict_1, stat_dict_2, label_1, label_2):
        """
        Safely divides values from two dictionaries, returns 0 if division by zero.

        Parameters:
        stat_dict_1 (dict): First dictionary containing statistics.
        stat_dict_2 (dict): Second dictionary containing statistics.
        label_1 (str): Key to extract from stat_dict_1.
        label_2 (str): Key to extract from stat_dict_2.

        Returns:
        float: Result of the division, or 0 if division by zero occurs.
        """
        return stat_dict_1[label_1].values[0] / stat_dict_2[label_2].values[0] if stat_dict_2[label_2].values[0] != 0 else 0

    def get_value(self, df, column):
        """
        Safely retrieves a value from a DataFrame column.

        Parameters:
        df (DataFrame): The DataFrame to extract from.
        column (str): The column name to retrieve.

        Returns:
        value: Value from the column or 0 if the DataFrame is empty.
        """
        return df[column].values[0] if not df.empty else 0

    def add_result_percentage(self, row, home_team_stats, away_team_stats):
        row['HT_Win%'] = self.divide(
            home_team_stats, home_team_stats, 'Wins', 'NumOfMatches')
        row['AT_Win%'] = self.divide(
            away_team_stats, away_team_stats, 'Wins', 'NumOfMatches')
        row['HT_Draw%'] = self.divide(
            home_team_stats, home_team_stats, 'Draws', 'NumOfMatches')
        row['AT_Draw%'] = self.divide(
            away_team_stats, away_team_stats, 'Draws', 'NumOfMatches')
        row['HT_Loss%'] = self.divide(
            home_team_stats, home_team_stats, 'Losses', 'NumOfMatches')
        row['AT_Loss%'] = self.divide(
            away_team_stats, away_team_stats, 'Losses', 'NumOfMatches')

        return row

    def add_possession_stats(self, row, home_team_stats, away_team_stats):
        row['HT_Possession%'] = self.divide(
            home_team_stats, home_team_stats, 'Possession', 'NumOfMatches')
        row['AT_Possession%'] = self.divide(
            away_team_stats, away_team_stats, 'Possession', 'NumOfMatches')
        row['HT_Possession'] = home_team_stats['Possession'].values[0]
        row['AT_Possession'] = away_team_stats['Possession'].values[0]

        return row

    def add_odds(self, row, original_row):
        row['B365H'] = original_row['B365H']
        row['B365A'] = original_row['B365A']
        row['B365D'] = original_row['B365D']

        return row

    def add_xg(self, row, original_row):
        row['HT_XG'] = original_row['PreHXG']
        row['AT_XG'] = original_row['PreAXG']

        return row

    def add_home_away_result_percentage(self, row, home_team_stats, away_team_stats):
        row['HT_HomeWin%'] = self.divide(
            home_team_stats, home_team_stats, 'HomeWins', 'NumOfHomeMatches')
        row['AT_AwayWin%'] = self.divide(
            away_team_stats, away_team_stats, 'AwayWins', 'NumOfAwayMatches')
        row['HT_HomeDraw%'] = self.divide(
            home_team_stats, home_team_stats, 'HomeDraws', 'NumOfHomeMatches')
        row['AT_AwayDraw%'] = self.divide(
            away_team_stats, away_team_stats, 'AwayDraws', 'NumOfAwayMatches')
        row['HT_HomeLoss%'] = self.divide(
            home_team_stats, home_team_stats, 'HomeLosses', 'NumOfHomeMatches')
        row['AT_AwayLoss%'] = self.divide(
            away_team_stats, away_team_stats, 'AwayLosses', 'NumOfAwayMatches')

        return row

    def add_shooting_stats(self, row, home_team_stats, away_team_stats):
        row['HT_ShotOnGoalPerMatch'] = self.divide(
            home_team_stats, home_team_stats, 'ShotsOnGoal', 'NumOfMatches')
        row['AT_ShotOnGoalPerMatch'] = self.divide(
            away_team_stats, away_team_stats, 'ShotsOnGoal', 'NumOfMatches')
        row['HT_ShotOnTargetPerMatch'] = self.divide(
            home_team_stats, home_team_stats, 'ShotsOnTarget', 'NumOfMatches')
        row['AT_ShotOnTargetPerMatch'] = self.divide(
            away_team_stats, away_team_stats, 'ShotsOnTarget', 'NumOfMatches')
        row['HT_ShotOnTargetAccuracy'] = self.divide(
            home_team_stats, home_team_stats, 'ShotsOnTarget', 'ShotsOnGoal')
        row['AT_ShotOnTargetAccuracy'] = self.divide(
            away_team_stats, away_team_stats, 'ShotsOnTarget', 'ShotsOnGoal')

        return row

    def add_goal_stats(self, row, home_team_stats, away_team_stats):
        row['HT_GoalAccuracy'] = self.divide(
            home_team_stats, home_team_stats, 'Goals', 'ShotsOnTarget')
        row['AT_GoalAccuracy'] = self.divide(
            away_team_stats, away_team_stats, 'Goals', 'ShotsOnTarget')
        row['HT_GoalsPerMatch'] = self.divide(
            home_team_stats, home_team_stats, 'Goals', 'NumOfMatches')
        row['AT_GoalsPerMatch'] = self.divide(
            away_team_stats, away_team_stats, 'Goals', 'NumOfMatches')
        row['HT_HalfTimeGoalsPerMatch'] = self.divide(
            home_team_stats, home_team_stats, 'HalfTimeGoals', 'NumOfMatches')
        row['AT_HalfTimeGoalsPerMatch'] = self.divide(
            away_team_stats, away_team_stats, 'HalfTimeGoals', 'NumOfMatches')

        return row

    def add_conceded_stats(self, row, home_team_stats, away_team_stats):
        row['HT_GoalsConcededPerMatch'] = self.divide(
            home_team_stats, home_team_stats, 'Conceded', 'NumOfMatches')
        row['AT_GoalsConcededPerMatch'] = self.divide(
            away_team_stats, away_team_stats, 'Conceded', 'NumOfMatches')

        return row

    def add_last_n_matches_stats(self, row, home_team_stats, away_team_stats):
        row['HT_GoalsLastNMatches'] = home_team_stats['GoalsLastNMatches'].values[0]
        row['AT_GoalsLastNMatches'] = away_team_stats['GoalsLastNMatches'].values[0]
        row['HT_GoalDiffLastNMatches'] = home_team_stats['GoalDiffLastNMatches'].values[0]
        row['AT_GoalDiffLastNMatches'] = away_team_stats['GoalDiffLastNMatches'].values[0]

        return row

    def add_win_streak(self, row, home_team_stats, away_team_stats):
        row['HT_WinStreak'] = home_team_stats['WinStreak'].values[0]
        row['AT_WinStreak'] = away_team_stats['WinStreak'].values[0]

        return row


class XTestConstructor(XTableConstructor):
    """
    Constructs a feature table (X_test) for the test set, using historical training data to compute team stats.
    """
    def __init__(self, df_test, df_train, team_list, **kwargs):
        super().__init__(**kwargs)
        self.test_data: pd.DataFrame = df_test
        self.train_data: pd.DataFrame = df_train
        self.X_test_table: pd.DataFrame = pd.DataFrame()
        self.team_list: list[str] = team_list

        # Compute individual team stats using the training data
        self.team_stats_calculator = IndividualTeamStats(self.train_data, self.team_list).compute()

    def construct_table(self) -> pd.DataFrame:
        """
        Constructs the test feature table by iterating over each row in the test set.

        Returns:
        DataFrame: The constructed test feature table.
        """
        for _, match_row in self.test_data.iterrows():
            home_team = match_row['HomeTeam']
            away_team = match_row['AwayTeam']

            home_team_stats = self.team_stats_calculator.loc[self.team_stats_calculator['Team'] == home_team]
            away_team_stats = self.team_stats_calculator.loc[self.team_stats_calculator['Team'] == away_team]

            feature_row: dict = self.construct_row(match_row, home_team, away_team, home_team_stats, away_team_stats)
            self.X_test_table = self.X_test_table._append(feature_row, ignore_index=True)

        return self.X_test_table


class XTrainConstructor(XTableConstructor):
    """
    Constructs a feature table (X_train) for the training set, using historical data to compute team stats.
    """
    def __init__(self, df_train, team_list, **kwargs):
        super().__init__(**kwargs)

        self.train_data: pd.DataFrame = df_train
        self.team_list: list[str] = team_list
        self.X_train_table: pd.DataFrame = pd.DataFrame()

    def construct_table(self) -> pd.DataFrame:
        """
        Constructs the training feature table by iterating over each row in the training set.

        Returns:
        DataFrame: The constructed training feature table.
        """
        individual_stats_manager = IndividualTeamStats(self.train_data, self.team_list)

        for _, match_row in self.train_data.iterrows():
            individual_stats = self.update_individual_stats(match_row, individual_stats_manager)

            home_team = match_row['HomeTeam']
            away_team = match_row['AwayTeam']

            home_team_stats = individual_stats.loc[individual_stats['Team'] == home_team]
            away_team_stats = individual_stats.loc[individual_stats['Team'] == away_team]

            feature_row: dict = self.construct_row(match_row, home_team, away_team, home_team_stats, away_team_stats)
            self.X_train_table = self.X_train_table._append(feature_row, ignore_index=True)

        return self.X_train_table

    def update_individual_stats(self, match_row, stats_manager: IndividualTeamStats) -> pd.DataFrame:
        """
        Updates individual team stats for a given match row.

        Parameters:
        match_row (Series): Row of match data.
        stats_manager (IndividualTeamStats): Manager for computing team stats.

        Returns:
        DataFrame: Updated DataFrame with new stats.
        """
        stats_manager.compute_team_stats(match_row)
        return stats_manager.generate_features_dataframe()