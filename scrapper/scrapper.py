import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
import pandas as pd

# The website which is being scrapping is simple 
# and the scrapper itself is supposed to be used once to get a dataset.
# So, I find it acceptable to not deal with exceptions processing, logging and etc.

root_url = "https://slova.org.ru/"

def clear_text(text: str):
    text = re.sub(r"<em>.*</em>", "", text)
    text = re.sub(r"<br/>", "\n", text)
    text = text.strip()
    
    return text

async def parse_single_poem(
    url: str,
    session: aiohttp.ClientSession
):
    poem_page = await session.get(root_url + url)
    bs4_page = BeautifulSoup(await poem_page.text(), "html.parser")
    
    if bs4_page.find("p"):
        poem_text = "".join(map(str, bs4_page.find("p").contents))
        poem_text = clear_text(poem_text)
    else:
        poem_text = bs4_page.find_all("pre")[-1].get_text()
    
    return poem_text, url.split("/")[1]

async def main():
    links = open("src/links.txt", "r").read().split()
    
    async with aiohttp.ClientSession() as session:
        authors_pages = [session.get(link) for link in links]
        results = await asyncio.gather(*authors_pages)
        
        poems_pages = []
        for result in results:
            bs4_page = BeautifulSoup(await result.text(), "html.parser")
            links_to_poems = bs4_page.find("main", class_="container").find("div", class_="grid-col-1").find_all("a")
            poems_pages.extend(map(lambda x: x["href"], links_to_poems))
        pages_parsing = [parse_single_poem(page, session) for page in poems_pages]
        results = await asyncio.gather(*pages_parsing)

        output = {"text": [], "label": []}
        for text, author in results:
            output["text"].append(text)
            output["label"].append(author)
        df_output = pd.DataFrame(output)
        df_output.to_csv("dataset/output.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())