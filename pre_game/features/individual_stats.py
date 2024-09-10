class IndividualTeamStats:
    """
    This class computes and stores individual team statistics for football matches, including wins, losses,
    goals, fouls, cards, possession, and other team-specific performance metrics.
    """
    def __init__(self, match_data: pd.DataFrame, team_list: list[str], num_last_matches=6):
        """
        Initialises the class with match data, a list of unique teams, and the number of last matches to consider.

        Parameters:
        match_data (DataFrame): The match dataset containing all relevant match statistics.
        team_list (list[str]): A list of unique team names.
        num_last_matches (int): The number of last matches to track for specific metrics (default is 6).
        """
        self.match_data = match_data
        self.reverse_data = self.match_data.iloc[::-1]  # Reverse the DataFrame to work with latest matches first
        self.team_list = team_list
        self.num_last_matches = num_last_matches

        # Initialise dictionaries to store team-specific statistics
        self.team_win_counts = self.init_teams_dict()
        self.team_foul_counts = self.init_teams_dict()
        self.team_draw_counts = self.init_teams_dict()
        self.team_goal_counts = self.init_teams_dict()
        self.team_loss_counts = self.init_teams_dict()
        self.team_seasons_played = self.init_teams_set()
        self.team_corner_counts = self.init_teams_dict()
        self.team_conceded_goals = self.init_teams_dict()
        self.team_red_card_counts = self.init_teams_dict()
        self.team_home_win_counts = self.init_teams_dict()
        self.team_away_win_counts = self.init_teams_dict()
        self.team_possession_stats = self.init_teams_dict()
        self.team_win_streaks = self.init_teams_dict()
        self.team_home_draw_counts = self.init_teams_dict()
        self.team_away_draw_counts = self.init_teams_dict()
        self.team_home_loss_counts = self.init_teams_dict()
        self.team_away_loss_counts = self.init_teams_dict()
        self.team_yellow_card_counts = self.init_teams_dict()
        self.team_shots_on_goal = self.init_teams_dict()
        self.team_shots_on_target = self.init_teams_dict()
        self.team_ht_goal_counts = self.init_teams_dict()
        self.team_last_n_goals = self.init_teams_dict_with_list()
        self.team_last_n_goal_diffs = self.init_teams_dict_with_list()

    def init_teams_dict(self) -> dict:
        """Initialises a dictionary for each team with zero values for integer-based statistics."""
        return {team: 0 for team in self.team_list}
    
    def init_teams_set(self) -> dict:
        """Initialises a dictionary for each team with empty sets for tracking unique values, such as seasons."""
        return {team: set() for team in self.team_list}
    
    def init_teams_dict_with_list(self) -> dict:
        """Initialises a dictionary for each team with empty lists for tracking recent match statistics."""
        return {team: [] for team in self.team_list}
    
    def compute_team_stats(self, match_row):
        """Calculates and updates the statistics for a given match row."""
        self.update_results(match_row)
        self.update_home_away_results(match_row)
        self.update_win_streaks(match_row)
        self.update_goals(match_row)
        self.update_conceded_goals(match_row)
        self.update_half_time_goals(match_row)
        self.update_shots_on_goal(match_row)
        self.update_shots_on_target(match_row)
        self.update_yellow_cards(match_row)
        self.update_red_cards(match_row)
        self.update_corners(match_row)
        self.update_possession(match_row)
        self.update_fouls(match_row)
        self.update_seasons(match_row)
        self.update_last_n_matches(match_row)

    def update_results(self, match_row):
        """Updates win, draw, and loss statistics for home and away teams based on the final result."""
        if match_row['FTR'] == "H":
            self.team_win_counts[match_row['HomeTeam']] += 1
            self.team_loss_counts[match_row['AwayTeam']] += 1
        elif match_row['FTR'] == "A":
            self.team_win_counts[match_row['AwayTeam']] += 1
            self.team_loss_counts[match_row['HomeTeam']] += 1
        elif match_row['FTR'] == "D":
            self.team_draw_counts[match_row['HomeTeam']] += 1
            self.team_draw_counts[match_row['AwayTeam']] += 1

    def update_home_away_results(self, match_row):
        """Updates home and away win, draw, and loss statistics."""
        if match_row['FTR'] == "H":
            self.team_home_win_counts[match_row['HomeTeam']] += 1
            self.team_away_loss_counts[match_row['AwayTeam']] += 1
        elif match_row['FTR'] == "A":
            self.team_away_win_counts[match_row['AwayTeam']] += 1
            self.team_home_loss_counts[match_row['HomeTeam']] += 1
        elif match_row['FTR'] == "D":
            self.team_home_draw_counts[match_row['HomeTeam']] += 1
            self.team_away_draw_counts[match_row['AwayTeam']] += 1
    
    def update_win_streaks(self, match_row):
        """Updates the win streak count for teams, resetting it on a draw or loss."""
        if match_row['FTR'] == "H":
            self.team_win_streaks[match_row['HomeTeam']] += 1
            self.team_win_streaks[match_row['AwayTeam']] = 0
        elif match_row['FTR'] == "A":
            self.team_win_streaks[match_row['AwayTeam']] += 1
            self.team_win_streaks[match_row['HomeTeam']] = 0
        elif match_row['FTR'] == "D":
            self.team_win_streaks[match_row['HomeTeam']] = 0
            self.team_win_streaks[match_row['AwayTeam']] = 0

    def update_goals(self, row):
        self.team_goals[row['HomeTeam']] += row['FTHG']
        self.team_goals[row['AwayTeam']] += row['FTAG']

    def update_conceded_goals(self, row):
        self.team_conceded[row['HomeTeam']] += row['FTAG']
        self.team_conceded[row['AwayTeam']] += row['FTHG']

    def update_half_time_goals(self, row):
        self.team_half_time_goals[row['HomeTeam']] += row['HTHG']
        self.team_half_time_goals[row['AwayTeam']] += row['HTAG']

    def update_shots_on_goal(self, row):
        self.team_shots_on_goal[row['HomeTeam']] += row['HS']
        self.team_shots_on_goal[row['AwayTeam']] += row['AS']

    def update_shots_on_target(self, row):
        self.team_shots_on_target[row['HomeTeam']] += row['HST']
        self.team_shots_on_target[row['AwayTeam']] += row['AST']

    def update_yellow_cards(self, row):
        self.team_yellow_cards[row['HomeTeam']] += row['HY']
        self.team_yellow_cards[row['AwayTeam']] += row['AY']

    def update_red_cards(self, row):
        self.team_red_cards[row['HomeTeam']] += row['HR']
        self.team_red_cards[row['AwayTeam']] += row['AR']

    def update_corners(self, row):
        self.team_corners[row['HomeTeam']] += row['HC']
        self.team_corners[row['AwayTeam']] += row['AC']

    def update_posession(self, row):
        self.team_possession[row['HomeTeam']] += row['HP']
        self.team_possession[row['AwayTeam']] += row['AP']

    def update_fouls(self, row):
        self.team_fouls[row['HomeTeam']] += row['HF']
        self.team_fouls[row['AwayTeam']] += row['AF']

    def update_seasons(self, row):
        self.team_seasons[row['HomeTeam']].add(row['Date'].year)
        self.team_seasons[row['AwayTeam']].add(row['Date'].year)

    def update_last_n_matches(self, row):
        if len(self.team_last_n_matches_goals[row['HomeTeam']]) >= self.last_n_matches:
            self.team_last_n_matches_goals[row['HomeTeam']].pop(0)
            self.team_last_n_matches_goal_diff[row['HomeTeam']].pop(0)

        if len(self.team_last_n_matches_goals[row['AwayTeam']]) >= self.last_n_matches:
            self.team_last_n_matches_goals[row['AwayTeam']].pop(0)
            self.team_last_n_matches_goal_diff[row['AwayTeam']].pop(0)

        self.team_last_n_matches_goals[row['HomeTeam']].append(row['FTHG'])
        self.team_last_n_matches_goals[row['AwayTeam']].append(row['FTAG'])
        self.team_last_n_matches_goal_diff[row['HomeTeam']].append(row['FTHG'] - row['FTAG'])
        self.team_last_n_matches_goal_diff[row['AwayTeam']].append(row['FTAG'] - row['FTHG']) 

    def generate_features_dataframe(self) -> pd.DataFrame:
        data = {
            'Team': [],
            'Wins': [],
            'Draws': [],
            'Losses': [],
            'HomeWins': [],
            'AwayWins': [],
            'HomeDraws': [],
            'AwayDraws': [],
            'HomeLosses': [],
            'AwayLosses': [],
            'Goals': [],
            'ShotsOnGoal': [],
            'ShotsOnTarget': [],
            'YellowCards': [],
            'RedCards': [],
            'Corners': [],
            'Fouls': [],
            'Seasons': [],
            'NumOfMatches': [],
            'NumOfHomeMatches': [],
            'NumOfAwayMatches': [],
            'Conceded': [],
            'HalfTimeGoals': [],
            'GoalsLastNMatches': [],
            'GoalDiffLastNMatches': [],
            'WinStreak': [],
            'Possession': [],
        }

        for team in self.unique_teams:
            data['Team'].append(team)
            data['Wins'].append(self.team_wins[team])
            data['Draws'].append(self.team_draws[team])
            data['Losses'].append(self.team_losses[team])
            data['HomeWins'].append(self.team_home_wins[team])
            data['AwayWins'].append(self.team_away_wins[team])
            data['HomeDraws'].append(self.team_home_draws[team])
            data['AwayDraws'].append(self.team_away_draws[team])
            data['HomeLosses'].append(self.team_home_losses[team])
            data['AwayLosses'].append(self.team_away_losses[team])
            data['Goals'].append(self.team_goals[team])
            data['ShotsOnGoal'].append(self.team_shots_on_goal[team])
            data['ShotsOnTarget'].append(self.team_shots_on_target[team])
            data['YellowCards'].append(self.team_yellow_cards[team])
            data['RedCards'].append(self.team_red_cards[team])
            data['Corners'].append(self.team_corners[team])
            data['Fouls'].append(self.team_fouls[team])
            data['Seasons'].append(len(self.team_seasons[team]))
            data['NumOfMatches'].append(self.team_wins[team] + self.team_draws[team] + self.team_losses[team])
            data['NumOfHomeMatches'].append(self.team_home_wins[team] + self.team_home_draws[team] + self.team_home_losses[team])
            data['NumOfAwayMatches'].append(self.team_away_wins[team] + self.team_away_draws[team] + self.team_away_losses[team])
            data['Conceded'].append(self.team_conceded[team])
            data['HalfTimeGoals'].append(self.team_half_time_goals[team])
            data['GoalsLastNMatches'].append(sum(self.team_last_n_matches_goals[team]))
            data['GoalDiffLastNMatches'].append(sum(self.team_last_n_matches_goal_diff[team]))
            data['WinStreak'].append(self.team_win_streak[team])
            data['Possession'].append(self.team_possession[team])

        return pd.DataFrame(data)
    
    def compute(self) -> pd.DataFrame:
        for _, row in self.reversed_df.iterrows():
            self.compute_team_stats(row)

        return self.generate_features_dataframe()