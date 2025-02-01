# Data

## Where is the Data used

The Data is needed for training/inference on every model. Currently, the code has the following models:
- INSEE France:
  - Scope: France only
  - Data source: INSEE (Institut National de la Statistique et des Etudes Economiques)
  - Data structure: Multiple tables, each containing 
  - Data processing:
    - At download: The data is downloaded table by table via API and split into a data DataFrame and a metadata DataFrame. We remove columns with no values before a certain date and after a certain date. Default arguments can be checked inside: `scripts.data_fetching.create_insee_tables.py`.
    - At train/inference: The data is processed using the DataProcessor, which further transforms tables for training/inference purpose.
  - License: Free for personal and commercial use. Website link: www.insee.fr
- CEPII World:
  - Scope: World wide
  - Data source: CEPII (Centre d'Etudes Prospectives et d'Informations Internationales)
  - Data structure:
    - BACI: A csv each year containing the trade flow per product between each pair of countries. A csv for mapping country ids to country ISO3 codes.
    - Gravity: A single csv containing all (CountryA, CountryB, Year) and different bilateral features (Conflict, Trade Agreement, Distance, ..)
  - Data processing:
    - At download: The data is downloaded manually, without further processing.
    - At train/inference: Relevant data are extracted. Potentially further refinement steps will be done via the DataProcessor object.
  - License: Etalab 2.0 (Any use is allowed provided explicit references). Website link: https://www.cepii.fr/CEPII/en/welcome.asp
- World Bank:
  - Score: World wide
  - Data source: World Bank Open Data
  - Data structure: Indicators are composed of multiple time series. Each csv file is a manually downloaded (faster than API for bulk data) csv file containing an indicator.
  - Data processing:
    - At download: The data is downloaded indicator by indicator, and series with little to no data for countries of interest are discarded.
    - At train/inference: TBD
  - License: Creative Commons Attribution 4.0 International License (Copy, redistribute, remix and build upon for personal and commercial use). Website link: https://data.worldbank.org/


## How to download the Data/Processed data

### Manual download
The raw data will not be provided in this code if a processing is required in order to standardize. Links for manually downloaded datasets and scripts for API datasets are present for you to download and process the data locally. See `scripts.data_fetching.*`

### Accessing Azure storage of standardized Data
In order to make this project more accessible, scripts are being rolled out to download the data locally directly from our Azure Storage, with some of the initial preprocessing already performed.
This is ideal if you want to quickly test the data, as API/website download speed can be quite slow and the datasets are multiple Gb.

In order to access the standardized data directly from our storage, follow these steps:
- Open a dicussion to request a readonly account, with the name you want and an email adress to communicate your password. We will create a `yourname@fondation.one` Azure account and send you your login details via Email.
- Download `Azure CLI` (Windows/Ubuntu process varies, so check the relevant installation on windows website for `Azure CLI`)
- Log in to your Azure account in the CLI by typing `az login` if you have access to a browser, or `az login --use-device-code` to log in via another machine with a code.
- Run scripts inside `scripts.data_fetching.xxx`
