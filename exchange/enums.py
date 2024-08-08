from enum import Enum
from metadata import HorseRacingMetadataBuilder, FootballMetadataBuilder, TennisMetadataBuilder

class MarketFilters(Enum):
    # ^ - start of the string, $ - end of the string
    FootballMarketRegex = r"(^MATCH_ODDS$)|(OVER)|(UNDER)|(_OU_)"
    TennisMarketRegex = r"(^MATCH_ODDS$)"
    HorseRacingMarketRegex = r"(^WIN$)|(^EACH_WAY$)"

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value
    

class MetaBuilder(Enum):
    Football = FootballMetadataBuilder
    Tennis = TennisMetadataBuilder
    HorseRacing = HorseRacingMetadataBuilder

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value
    

class Sport(Enum):
    Football = "football"
    Tennis = "tennis"
    HorseRacing = "horseracing"

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value
    

class CountryFilters(Enum):
    FootballCountryRegex = r".*"
    HorseRacingCountryRegex = r"(GB)|(IE)"
    TennisCountryRegex = r".*"

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value
    

class Collections(Enum):
    Metadata = "metadata"
    Ladders = "ladders"
    Marketdata = "marketdata"

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value
    

class Databases(Enum):
    Football = "football_betfair"
    Tennis = "tennis_betfair"
    HorseRacing = "horseracing_betfair"
    Betfair = "betfair_exchange"

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value


class MongoURIs(Enum):
    Test = "mongodb+srv://benschlagman:Neh6x0VGiCo1jnIX@bf-exchange-test-01.tfbzy1n.mongodb.net/?retryWrites=true&w=majority"
    Serverless = "mongodb+srv://matyashuba:qXvE54CiMELUbUMi@bf-exchange-01.2h7jxvl.mongodb.net/?retryWrites=true&w=majority&appName=bf-exchange-01"

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value


class AWS(Enum):
    Key = 'AKIAZXVKJEGNJYP2L2EC'
    Secret = '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'
    Bucket = 'historicdata-qst'

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value
    