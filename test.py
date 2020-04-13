import requests

URL = "https://opendart.fss.or.kr/api/list.json"
KEY = "f90ae56d6e8a413a7cbc5c86a50ee97d2fe4c413"
res = requests.get(URL)
print(res.status_code)

parameters = {'crtfc_key': KEY, 'corp_code': '005930'}
res = requests.get(URL, params=parameters)
print(res.text)

import FinanceDataReader as fdr
df = fdr.DataReader('263750')
print(df)

from korea_news_crawler.articlecrawler import ArticleCrawler

if __name__ == '__main__':
    Crawler = ArticleCrawler()
    Crawler.set_category("정치")
    Crawler.set_date_range(2017, 1, 2018, 1)
    Crawler.start()