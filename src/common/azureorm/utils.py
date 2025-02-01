import configparser


class OrmUtils:
    ini_config: configparser.ConfigParser = None

    @staticmethod
    def load_ini_config() -> configparser.ConfigParser:
        """
        Load the appsettings.ini configuration file.
        This file should be located at the root of the project.
        appsettings.ini is a configuration file that should only contain non-sensitive information (paths, urls, etc.)

        :return: The config file, which you can access like a dictionary. Example: config["section"]["key"]
        """
        if OrmUtils.ini_config is not None:
            return OrmUtils.ini_config

        config = configparser.ConfigParser()
        config.read("appsettings.ini")

        OrmUtils.ini_config = config
        return config