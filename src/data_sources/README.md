# Data sources

## Purpose

This folder contains all the classes that are responsible for loading and processing the raw data into a clean format that can be exported to a csv file or in a database.

## Structure
The structure always follows the same pattern:
- A fetcher class that is responsible for loading the raw csv file into a pandas DataFrame.
- A pipeline that is responsible for cleaning and processing the data.
- (Optional) A utility method that can be used by other classes to process their own data. For example inflation data is used to adjust values of other datasets.