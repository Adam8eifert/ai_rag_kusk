import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://zakazky.kr-stredocesky.cz/"
INDEX_URL = "https://zakazky.kr-stredocesky.cz/agreement_index.html"
OUTPUT_DIR = "data"
MAX_FILES = 10


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    response = requests.get(INDEX_URL, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    pdf_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".pdf"):
            pdf_links.append(urljoin(BASE_URL, href))

    print(f"Nalezeno PDF: {len(pdf_links)}")

    for i, pdf_url in enumerate(pdf_links[:MAX_FILES], start=1):
        print(f"Stahuji {i}/{MAX_FILES}: {pdf_url}")

        pdf_response = requests.get(pdf_url, timeout=20)
        pdf_response.raise_for_status()

        filename = os.path.join(OUTPUT_DIR, f"smlouva_{i:02d}.pdf")
        with open(filename, "wb") as f:
            f.write(pdf_response.content)

    print("Hotovo.")


if __name__ == "__main__":
    main()
