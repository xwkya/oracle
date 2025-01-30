import configparser
from pathlib import Path


class CoreUtils:
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
        config = configparser.ConfigParser()
        config.read("appsettings.ini")
        return config
