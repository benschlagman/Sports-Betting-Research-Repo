from data.footystats import GenerateDataFrame
from utils.data_utils import Season
from utils.model_utils import Feature

import model


if __name__ == '__main__':
    season = Season.Past1
    df = GenerateDataFrame(season=season).load()

    params = {
        Feature.GOAL_STATS.value: True,
        Feature.SHOOTING_STATS.value: True,
        Feature.POSSESSION_STATS.value: True,
        Feature.ODDS.value: True,
        Feature.XG.value: True,
        Feature.HOME_AWAY_RESULTS.value: True,
        Feature.CONCEDED_STATS.value: True,
        Feature.LAST_N_MATCHES.value: True,
        Feature.WIN_STREAK.value: True,

    }

    model.run(df=df, feature_params=params)
