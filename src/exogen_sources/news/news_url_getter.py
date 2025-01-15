from random import sample

import requests
from dataclasses import dataclass
from typing import List
from gnews import GNews
from urllib.parse import urlparse, urljoin
import datetime
import dateparser
from googlenewsdecoder import new_decoderv1

@dataclass
class NewsUrl:
    url: str
    date: datetime.datetime

def decode_url(url: str):
    """
    Decodes a URL using new_decoderv1 synchronously.
    Returns the decoded URL if successful, None otherwise.
    """
    try:
        decoded = new_decoderv1(url)
        if decoded.get("status"):
            return decoded["decoded_url"]
        else:
            print("Error decoding URL:", decoded.get("message"))
            return None
    except Exception as e:
        print(f"Error occurred while decoding {url}: {e}")
        return None

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
        urls = [NewsUrl(article['url'], date=dateparser.parse(article['published date']))
                for article in news]
        extracted_urls = []

        for newsurl in urls:
            url = newsurl.url
            decoded_url = decode_url(url)
            if decoded_url:
                extracted_urls.append(NewsUrl(url=decoded_url, date=newsurl.date))

        return extracted_urls


class NYTNewsGetter:
    def __init__(self, language="en", country="US", max_results=100, start_date=None, end_date=None,
                 exclude_websites=None, proxy=None):
        """
        Maintains the same signature as NewsUrlGetter for compatibility
        """
        self.max_results = max_results
        self.api_key = "YOivS215fIOG2zzmAhAAJ1c5EXNhrwur"
        self.start_date = start_date
        self.end_date = end_date

    def get_news_url(self, topic: str, timeout=2) -> List[NewsUrl]:
        """
        Get news URLs from NYT Archive API
        """
        start_year = self.start_date[0]
        start_month = str(int(self.start_date[1]))
        url = f"https://api.nytimes.com/svc/archive/v1/{start_year}/{start_month}.json?api-key={self.api_key}"

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            articles = response.json()['response']['docs']
            articles = [art for art in articles if art['document_type'] == 'article']
            filtered_articles = [art for art in articles if 'print_page' in art and int(art['print_page']) < 3]

            if len(filtered_articles) > self.max_results > 0:
                filtered_articles = sample(filtered_articles, self.max_results)

            return [NewsUrl(
                url=article['web_url'],
                date=dateparser.parse(article['pub_date'])
            ) for article in filtered_articles]

        except requests.exceptions.RequestException as e:
            print(f"Error fetching from NYT API: {e}")
            return []