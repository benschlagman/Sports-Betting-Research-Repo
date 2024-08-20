from data.footystats import GenerateDataFrame
from utils.data_utils import Season
import model


if __name__ == '__main__':
    season = Season.Past5
    df = GenerateDataFrame(season=season).load()
    model.run(df=df)
