import requests
import pandas as pd
from data.clean import FootyStatsCleaner
from utils.data_utils import Season

API_KEY = "add_key"

class APIClient:
    """Football data API client that handles requests to the football data API"""

    def __init__(self, api_key):
        """
        Initialise the API client with the provided API key.
        
        Parameters:
        api_key (str): API key for authentication.
        """
        self.api_key = api_key
        self.BASE_URL = 'https://api.football-data-api.com'

    def _make_request(self, endpoint, params=None):
        """
        A general method to make API requests to specified endpoints.
        
        Parameters:
        endpoint (str): The API endpoint to request.
        params (dict): Additional parameters for the request.
        
        Returns:
        dict: The parsed JSON response from the API.
        """
        params = params or {}
        params = {'key': self.api_key, **params}
        response = requests.get(f'{self.BASE_URL}/{endpoint}', params=params)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()['data']


class LeagueDataClient(APIClient):
    """API client for fetching league-related data from the football API"""

    def get_league_list(self, **kwargs):
        """Retrieve a list of available leagues"""
        return self._make_request('league-list', params=kwargs)

    def get_country_list(self):
        """Retrieve a list of countries"""
        return self._make_request('country-list')

    def get_todays_matches(self, **kwargs):
        """Retrieve today's matches"""
        return self._make_request('todays-matches', params=kwargs)

    def get_league_matches(self, season_id, **kwargs):
        """Retrieve matches for a specific league season"""
        kwargs.update({'season_id': season_id})
        return self._make_request('league-matches', params=kwargs)

    def get_league_season(self, season_id, **kwargs):
        """Retrieve season details for a league"""
        kwargs.update({'season_id': season_id})
        return self._make_request('league-season', params=kwargs)

    def get_league_teams(self, season_id, **kwargs):
        """Retrieve teams for a specific league season"""
        kwargs.update({'season_id': season_id})
        return self._make_request('league-teams', params=kwargs)

    def get_league_players(self, season_id, **kwargs):
        """Retrieve players for a specific league season"""
        kwargs.update({'season_id': season_id})
        return self._make_request('league-players', params=kwargs)

    def get_league_referees(self, season_id, **kwargs):
        """Retrieve referees for a specific league season"""
        kwargs.update({'season_id': season_id})
        return self._make_request('league-referees', params=kwargs)

    def get_team(self, team_id):
        """Retrieve details for a specific team"""
        return self._make_request('team', {'team_id': team_id})

    def get_lastx(self, team_id):
        """Retrieve last 'x' matches for a specific team"""
        return self._make_request('lastx', {'team_id': team_id})

    def get_match_stats(self, match_id):
        """Retrieve statistics for a specific match"""
        return self._make_request('match', {'match_id': match_id})


class GenerateDataFrame():
    """Class for generating DataFrame objects from football API data"""

    def __init__(self, league_name="Premier League", country="England", season: Season = Season.Past1):
        """
        Initialise the data generator with league, country, and season.
        
        Parameters:
        league_name (str): The name of the league.
        country (str): The country where the league is based.
        season (Season): The season for which data is being generated.
        """
        self.client = LeagueDataClient(API_KEY)
        self.league_name = league_name
        self.country = country
        self.years = self.get_years(season)

    def get_years(self, season: Season):
        """
        Calculate the range of years based on the season.
        
        Parameters:
        season (Season): The season object representing the starting year.
        
        Returns:
        list: A list of years as strings.
        """
        first_season_year = season.value.year
        last_season_year = 2023
        return [str(year) for year in range(first_season_year, last_season_year + 1)]

    def get_league_list_df(self):
        """Retrieve the list of leagues as a DataFrame"""
        league_data = self.client.get_league_list()
        leagues_df = pd.DataFrame(league_data)
        return leagues_df

    def get_matches_by_league_df(self, season_id, max_per_page=None, page=None, max_time=None):
        """
        Retrieve league matches for a specific season as a DataFrame.
        
        Parameters:
        season_id (int): The ID of the season.
        max_per_page (int): Maximum number of results per page.
        page (int): Page number for pagination.
        max_time (int): Maximum time for match retrieval.
        
        Returns:
        DataFrame: A DataFrame of league matches.
        """
        params = {'max_per_page': max_per_page, 'page': page, 'max_time': max_time}
        match_data = self.client.get_league_matches(season_id, **params)
        return pd.DataFrame(match_data)

    def get_filtered_leagues(self, league_name, country, years):
        """
        Filter the list of leagues based on the league name, country, and years.
        
        Parameters:
        league_name (str): Name of the league to filter.
        country (str): Country of the league.
        years (list): List of years to filter the leagues by.
        
        Returns:
        DataFrame: Filtered DataFrame of league seasons.
        """
        leagues_df = self.get_league_list_df()
        filtered_leagues = leagues_df[(leagues_df['country'] == country) & (leagues_df['league_name'] == league_name)]

        league_seasons_data = []

        for _, row in filtered_leagues.iterrows():
            for season in row['season']:
                season_data = {
                    'id': season['id'],
                    'year': str(season['year'])[:4],
                    'league_name': row['league_name'],
                    'country': row['country']
                }
                league_seasons_data.append(season_data)

        league_seasons_df = pd.DataFrame(league_seasons_data)
        return league_seasons_df[league_seasons_df['year'].isin(years)]

    def get_footystats_matches(self, league_name, country, years):
        """
        Retrieve match data from the filtered leagues.
        
        Parameters:
        league_name (str): Name of the league.
        country (str): Country of the league.
        years (list): List of years to filter the data by.
        
        Returns:
        DataFrame: A DataFrame of matches for the filtered leagues.
        """
        filtered_leagues = self.get_filtered_leagues(league_name, country, years)
        all_matches = pd.DataFrame()

        for _, row in filtered_leagues.iterrows():
            matches_df = self.get_matches_by_league_df(row['id'])
            matches_df['season_id'] = row['id']
            matches_df['league_name'] = row['league_name']
            matches_df['country'] = row['country']
            matches_df['year'] = row['year']
            all_matches = pd.concat([all_matches, matches_df], ignore_index=True)

        return all_matches

    def get_referee_map_for_seasons(self, season_ids):
        """Retrieve a mapping of referee IDs to names for a list of seasons"""
        referee_map = {}
        for season_id in season_ids:
            referees = self.client.get_league_referees(season_id)
            for ref in referees:
                referee_map[ref['id']] = ref['full_name']
        return referee_map

    def get_team_images_for_ids(self, team_ids):
        """Retrieve team logos for a list of team IDs"""
        team_image_map = {}
        for team_id in team_ids:
            team_data = self.client.get_team(team_id)
            if team_data and 'image' in team_data[0]:
                team_image_map[team_id] = team_data[0]['image']
        return team_image_map

    def load(self, save_to_file=False) -> pd.DataFrame:
        """
        Load the match data into a DataFrame and optionally save it to a CSV file.
        
        Parameters:
        save_to_file (bool): Whether to save the data to a CSV file.
        
        Returns:
        DataFrame: The cleaned DataFrame after applying FootyStatsCleaner.
        """
        match_data_df = self.get_footystats_matches(self.league_name, self.country, self.years)
        if save_to_file:
            match_data_df.to_csv('raw_footystats.csv', index=False)
        return FootyStatsCleaner(match_data_df).run()
