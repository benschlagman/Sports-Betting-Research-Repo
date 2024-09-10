from enum import Enum
from metadata import HorseRacingMetadataBuilder, FootballMetadataBuilder, TennisMetadataBuilder


class MarketFilters(Enum):
    # ^ - start of the string, $ - end of the string
    # FootballMarketRegex = r"(^MATCH_ODDS$)|(OVER)|(UNDER)|(_OU_)"
    FootballMarketRegex = r"(^MATCH_ODDS$)"
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
    FootballCountryRegex = r"(GB)"
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
    Test = "mongodb+srv://username:password@bf-exchange-test-01.tfbzy1n.mongodb.net/?retryWrites=true&w=majority"
    Serverless = "mongodb+srv://username:password@bf-exchange-01.2h7jxvl.mongodb.net/?retryWrites=true&w=majority&appName=bf-exchange-01"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class AWS(Enum):
    Key = 'your_aws_access_key'
    Secret = 'your_aws_secret_key'
    Bucket = 'your_s3_bucket_name'

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
