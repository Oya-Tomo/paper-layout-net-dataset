import re
import xml.etree.ElementTree as ET


def xml_to_json(xml: ET.Element):
    # Convert XML to JSON
    def expand_children(node: ET.Element):
        obj = {
            "tag": node.tag.split("}")[1],
            "tag_prefix": re.findall(r"{.*}", node.tag)[0][1:-1],
            "text": node.text,
            "attributes": node.attrib,
            "children": [],
        }

        for child in node:
            obj["children"] = obj.get("children", []) + [expand_children(child)]
        return obj

    return expand_children(xml)


def xml_to_arxiv_json(xml: ET.Element):
    data = xml_to_json(xml)

    if data["tag"] != "feed":
        raise ValueError("Expected root tag to be 'feed'")

    entries = []

    for child in data["children"]:
        if child["tag"] == "entry":
            entry = {
                "id": None,
                "title": None,
                "summary": None,
                "updated": None,
                "published": None,
                "authors": [],
                "pdf": None,
                "comment": None,
                "journal_ref": None,
                "doi": None,
                "primary_category": None,
                "categories": [],
            }
            for entry_child in child["children"]:
                if entry_child["tag"] == "id":
                    entry["id"] = entry_child["text"]
                elif entry_child["tag"] == "title":
                    entry["title"] = entry_child["text"]
                elif entry_child["tag"] == "summary":
                    entry["summary"] = entry_child["text"]
                elif entry_child["tag"] == "updated":
                    entry["updated"] = entry_child["text"]
                elif entry_child["tag"] == "published":
                    entry["published"] = entry_child["text"]
                elif entry_child["tag"] == "author":
                    author = {
                        "name": None,
                        "affiliation": None,
                    }
                    for author_child in entry_child["children"]:
                        if author_child["tag"] == "name":
                            author["name"] = author_child["text"]
                        elif author_child["tag"] == "affiliation":
                            author["affiliation"] = author_child["text"]
                    entry["authors"].append(author)
                elif entry_child["tag"] == "link":
                    if entry_child["attributes"].get("type", None) == "application/pdf":
                        entry["pdf"] = entry_child["attributes"]["href"]
                elif entry_child["tag"] == "comment":
                    entry["comment"] = entry_child["text"]
                elif entry_child["tag"] == "journal_ref":
                    entry["journal_ref"] = entry_child["text"]
                elif entry_child["tag"] == "doi":
                    entry["doi"] = entry_child["text"]
                elif entry_child["tag"] == "primary_category":
                    entry["primary_category"] = entry_child["attributes"]["term"]
                elif entry_child["tag"] == "category":
                    entry["categories"].append(entry_child["attributes"]["term"])
            entries.append(entry)
    return entries


def url_to_arxiv_id(url: str) -> str | None:
    id_part = re.match(r"^(http|https)://arxiv.org/abs/([a-zA-Z0-9.\-/]*)$", url)
    if id_part is None:
        return None
    version_part = re.search(r"v\d+$", id_part.group(2))
    if version_part is None:
        return id_part.group(2)
    else:
        return id_part.group(2)[: -len(version_part.group(0))]


if __name__ == "__main__":
    import json

    import requests

    url = "http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=10"
    response = requests.get(url)
    xml = ET.fromstring(response.text)
    data = xml_to_arxiv_json(xml)
    print(json.dumps(data, indent=2))

    new_type_url = "https://arxiv.org/abs/2308.08241"
    print("old type arxiv id:", url_to_arxiv_id(new_type_url))
    old_type_url = "https://arxiv.org/abs/cond-mat/0102536v132"
    print("new type arxiv id:", url_to_arxiv_id(old_type_url))
