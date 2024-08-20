from enum import Enum


class Feature(Enum):
    GOAL_STATS = 'is_goal_stats'
    SHOOTING_STATS = 'is_shooting_stats'
    POSSESSION_STATS = 'is_possession_stats'
    RESULT = 'is_result'
    ODDS = 'is_odds'
    XG = 'is_xg'
    HOME_AWAY_RESULTS = 'is_home_away_results'
    CONCEDED_STATS = 'is_conceded_stats'
    LAST_N_MATCHES = 'is_last_n_matches'
    WIN_STREAK = 'is_win_streak'
