from data.footystats import GenerateDataFrame
from utils.data_utils import Season
import model

if __name__ == '__main__':
    # Set the desired season for data generation (e.g., past 5 seasons)
    selected_season = Season.Past5
    
    # Generate the DataFrame using the selected season
    data_frame = GenerateDataFrame(season=selected_season).load()
    
    # Run the model using the generated DataFrame
    model.run(df=data_frame)
