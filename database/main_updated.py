# Changed for exp3

from exchange.ladder import LadderBuilder
from interface.s3 import S3
from interface.mongo import MongoDB, GridFs
from exchange import betfair_utils
from exchange.enums import MarketFilters, Sport, CountryFilters, Collections, Databases, MongoURIs, MetaBuilder
import concurrent.futures
import multiprocessing
import threading


def run(s3: S3, mongodb: MongoDB, grid_fs: GridFs, file_key: str, market_filter: MarketFilters, meta_builder: MetaBuilder, country_filter: CountryFilters, stop_event: threading.Event):
    """
    Run the pipeline for a single file retrieved from the specified folder in S3.
    """

    if stop_event.is_set():
        return

    if not betfair_utils.is_market_file(file_key):
        return
    market_string_updates: list[str] = s3.get_file_content(file_key)
    if not market_string_updates:
        return

    marketdata: list[dict] = betfair_utils.json_load_updates(
        market_string_updates)
    market_definition: dict = betfair_utils.get_market_definition(marketdata)

    if not betfair_utils.is_matching_filters(market_definition, market_filter, country_filter):
        return
    metadata, ladders, ts_marketdata = LadderBuilder(
        marketdata, market_definition, meta_builder).run()

    print(
        f"Finished handling '{file_key} with thread id: {threading.current_thread().ident}'\n")

    return ladders


def main(folder: str, uri: MongoURIs, market_filter: MarketFilters, meta_builder: MetaBuilder, country_filter: CountryFilters, database: Databases, is_multiprocess: bool, max_results: int = 5):
    s3_interface = S3(folder)
    mongo_interface = MongoDB(database, uri)
    grid_fs_interface = GridFs(mongo_interface.db, Collections.Marketdata)

    all_ladders = []
    stop_event = threading.Event()

    if not is_multiprocess:
        for file_key in s3_interface.fetch_files_from_s3():
            ladders = run(
                s3=s3_interface,
                mongodb=mongo_interface,
                grid_fs=grid_fs_interface,
                file_key=file_key,
                market_filter=market_filter,
                meta_builder=meta_builder,
                country_filter=country_filter,
                stop_event=stop_event
            )

            if ladders:
                all_ladders.append(ladders)
            if len(all_ladders) >= max_results:
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for file_key in s3_interface.fetch_files_from_s3():
                future = executor.submit(
                    run,
                    s3=s3_interface,
                    mongodb=mongo_interface,
                    grid_fs=grid_fs_interface,
                    file_key=file_key,
                    market_filter=market_filter,
                    meta_builder=meta_builder,
                    country_filter=country_filter,
                    stop_event=stop_event
                )
                futures.append(future)

            try:
                for future in concurrent.futures.as_completed(futures):
                    ladders = future.result()
                    if ladders:
                        all_ladders.append(ladders)
                        if len(all_ladders) >= max_results:
                            stop_event.set()  # Signal other threads to stop
                            break
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                # Cancel remaining futures
                for future in futures:
                    if not future.done():
                        future.cancel()

    return all_ladders


if __name__ == "__main__":
    # Specify these parameters manually
    # ==========================================================
    folder = 'Soccer/PRO/2023/Jan/1/'
    sport = Sport.Football
    uri = MongoURIs.Serverless
    is_multiprocess = True  # Set to True for multithreading
    max_results = 5  # Stop after this many results
    # ==========================================================

    if sport == Sport.HorseRacing:
        main(
            folder,
            uri,
            MarketFilters.HorseRacingMarketRegex,
            MetaBuilder.HorseRacing,
            CountryFilters.HorseRacingCountryRegex,
            Databases.HorseRacing,
            is_multiprocess,
            max_results
        )
    elif sport == Sport.Football:
        main(
            folder,
            uri,
            MarketFilters.FootballMarketRegex,
            MetaBuilder.Football,
            CountryFilters.FootballCountryRegex,
            Databases.Football,
            is_multiprocess,
            max_results
        )
    elif sport == Sport.Tennis:
        main(
            folder,
            uri,
            MarketFilters.TennisMarketRegex,
            MetaBuilder.Tennis,
            CountryFilters.TennisCountryRegex,
            Databases.Tennis,
            is_multiprocess,
            max_results
        )
