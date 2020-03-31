import requests
import re
from bs4 import BeautifulSoup

html = requests.get('http://58921.com/alltime/2020')
html.encoding = 'utf-8'
html = html.text
# print(html)

req = r'<td><a href="/film/.*?" title="(.*?)">.*?</a></td>'
title = re.findall(req, html)
# print(title)

soup = BeautifulSoup(html, 'lxml')
img_url = soup.find_all('img')[1:]
for url in img_url:
    img_src = url.get('src')
    print(img_src)



