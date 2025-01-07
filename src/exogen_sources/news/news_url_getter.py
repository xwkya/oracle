import requests
from dataclasses import dataclass
from typing import List
from gnews import GNews
from urllib.parse import urlparse, urljoin
import datetime
import dateparser

@dataclass
class NewsUrl:
    url: str
    html: str
    date: datetime.datetime


class NewsUrlGetter(GNews):
    def __init__(self, language="en", country="US", max_results=100, start_date=None, end_date=None,
                 exclude_websites=None, proxy=None):
        """
        (optional parameters)
        :param language: The language in which to return results, defaults to en (optional)
        :param country: The country code of the country you want to get headlines for, defaults to US
        :param max_results: The maximum number of results to return. The default is 100, defaults to 100
        :param period: The period of time from which you want the news
        :param start_date: Date after which results must have been published
        :param end_date: Date before which results must have been published
        :param exclude_websites: A list of strings that indicate websites to exclude from results
        :param proxy: The proxy parameter is a dictionary with a single key-value pair. The key is the
        protocol name and the value is the proxy address
        """
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
            'Referer': 'https://www.google.com/'
        }

        super().__init__(language=language, country=country, start_date=start_date, end_date=end_date, max_results=max_results, exclude_websites=exclude_websites, proxy=proxy)

    @staticmethod
    def clean_url(url: str) -> str:
        """
        Clean a URL
        :param url: The URL to clean
        :return: The cleaned URL
        """

        return urljoin(url, urlparse(url).path)

    def get_news_url(self, topic: str, timeout=2) -> List[NewsUrl]:
        news = self.get_news(topic)
        urls = [NewsUrl(article['url'], None, date=dateparser.parse(article['published date']))
                for article in news]
        extracted_urls = []

        for newsurl in urls:
            url = newsurl.url
            try:
                # Allow redirects and get the final URL
                r = requests.get(url, timeout=timeout, headers=self.headers, allow_redirects=True)
                # The final URL after all redirects is in r.url
                newsurl.url = self.clean_url(r.url)
                newsurl.html = r.text
                extracted_urls.append(newsurl)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching URL {url}: {e}")
                continue

        return extracted_urls