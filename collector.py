import datetime
import hashlib
import xml.etree.ElementTree as ET
from typing import Self

import fitz
import requests
from PIL import Image
from pydantic import BaseModel

from arxiv import url_to_arxiv_id, xml_to_arxiv_json


class ArxivPaper(BaseModel):
    id: str
    src: str = "arxiv"

    title: str
    abstract: str

    authors: list[str]
    organizations: list[str]

    url: str
    pdf: str

    journal: str | None
    doi: str | None

    published_at: datetime.datetime
    updated_at: datetime.datetime

    @classmethod
    def from_dict(self, data: dict) -> Self:
        return ArxivPaper(
            id=url_to_arxiv_id(data["id"]),
            title=data["title"],
            abstract=data["summary"],
            authors=[author["name"] for author in data["authors"]],
            organizations=list(
                set(author["affiliation"] for author in data["authors"]) - {None}
            ),
            url=data["id"],
            pdf=data["pdf"],
            journal=data["journal_ref"],
            doi=data["doi"],
            published_at=datetime.datetime.fromisoformat(data["published"]),
            updated_at=datetime.datetime.fromisoformat(data["updated"]),
        )


def collect_arxiv_papers(
    search_query: str | None = None,
    id_list: str | None = None,
    start: int = 0,
    max_results: int = 10,
) -> list[ArxivPaper]:
    base_url = f"http://export.arxiv.org/api/query?"
    params = []
    if search_query is not None:
        params.append(f"search_query={search_query}")
    if id_list is not None:
        params.append(f"id_list={id_list}")
    params.append(f"start={start}")
    params.append(f"max_results={max_results}")
    url = base_url + "&".join(params)

    response = requests.get(url)
    xml = ET.fromstring(response.text)
    data = xml_to_arxiv_json(xml)

    return [ArxivPaper.from_dict(paper) for paper in data]


def download_pdf(url: str, path: str) -> bool:
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"Failed to download PDF: {response.status_code}")
        return False


def generate_short_hash(seed: str) -> str:
    hash = hashlib.md5(seed.encode()).hexdigest()
    return hash


def get_pdf_page_images(pdf_path: str, dpi: int = 300) -> list[Image.Image]:
    doc = fitz.open(pdf_path)
    images = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=dpi)
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return images


if __name__ == "__main__":
    import hashlib

    query = "cat:cs.CV"
    id_list = None
    start = 0
    max_results = 10
    papers = collect_arxiv_papers(
        search_query=query,
        id_list=id_list,
        start=start,
        max_results=max_results,
    )

    for paper in papers:
        pdf_path = f"cache/{generate_short_hash(paper.id)}.pdf"
        download_pdf(paper.pdf, pdf_path)
        images = get_pdf_page_images(pdf_path, dpi=150)
        for i, img in enumerate(images):
            img.save(f"cache/{generate_short_hash(paper.id)}_page_{i+1:03d}.png")
