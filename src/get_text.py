from urllib.request import urlopen
from bs4 import BeautifulSoup
articleURL = "http://curia.europa.eu/juris/document/document.jsf?text=&docid=139407&pageIndex=0&doclang=EN&mode=lst&dir=&occ=first&part=1&cid=52454"


def getText(url):
    page = urlopen(url).read().decode('utf8', 'ignore')
    soup = BeautifulSoup(page, 'lxml')
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return text.encode('ascii', errors='replace').decode().replace("?","")
text = getText(articleURL)