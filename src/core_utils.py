import configparser
from pathlib import Path

import pandas as pd

from azureorm import ORMWrapper


class CoreUtils:
    ini_config: configparser.ConfigParser = None

    @staticmethod
    def get_countries_of_interest() -> set:
        """
        Get the set of countries of interest for the project.
        """
        config = CoreUtils.load_ini_config()
        return set(pd.read_csv(config['datasets']['CountriesOfInterest'])['id'])

    @staticmethod
    def get_root() -> Path:
        """
        Returns a reference to the root path of the project.
        TODO: Cache this?
        """
        path = Path(__file__).resolve().parent
        while '.env' not in [f.name for f in path.iterdir()]:
            path = path.parent

        return path

    @staticmethod
    def load_ini_config() -> configparser.ConfigParser:
        """
        Load the appsettings.ini configuration file.
        This file should be located at the root of the project.
        appsettings.ini is a configuration file that should only contain non-sensitive information (paths, urls, etc.)

        :return: The config file, which you can access like a dictionary. Example: config["section"]["key"]
        """
        if CoreUtils.ini_config is not None:
            return CoreUtils.ini_config

        config = configparser.ConfigParser()
        config.read("appsettings.ini")

        CoreUtils.ini_config = config
        return config

    @staticmethod
    def get_orm() -> ORMWrapper:
        """
        Get the ORMWrapper instance for the project.
        """
        config = CoreUtils.load_ini_config()
        return ORMWrapper(config["database"]["db_server"], config["database"]["db_name"], config["database"]["db_port"])