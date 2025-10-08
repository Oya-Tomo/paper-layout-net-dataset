import os
import random

import torch
import yaml
from PIL import Image
from ultralytics.data.annotator import auto_annotate

from collector import (
    collect_arxiv_papers,
    download_pdf,
    generate_short_hash,
    get_pdf_page_images,
)
from model import load_yolo_model


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/images/test", exist_ok=True)

    os.makedirs("dataset/labels/train", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)
    os.makedirs("dataset/labels/test", exist_ok=True)

    with open("dataset/data.yaml", "w+") as data_yaml:
        yaml.dump(
            {
                "path": "./dataset",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {
                    0: "Caption",
                    1: "Footnote",
                    2: "Formula",
                    3: "List-item",
                    4: "Page-footer",
                    5: "Page-header",
                    6: "Picture",
                    7: "Section-header",
                    8: "Table",
                    9: "Text",
                    10: "Title",
                },
            },
            data_yaml,
            sort_keys=False,
        )

    model = load_yolo_model()

    queries = [
        "cat:cs.SC",
        "cat:cs.AI",
        "cat:cs.LG",
        "cat:cs.DC",
        "cat:cs.CL",
        "cat:cs.SE",
        "cat:cs.MA",
        "cat:cs.CR",
        "cat:cs.CE",
        "cat:cs.CV",
        "cat:cs.RO",
        "cat:cs.CY",
        "cat:cs.SD",
        "cat:cs.HC",
        "cat:cs.NI",
        "cat:cs.SY",
        "cat:cs.PL",
        "cat:math.NA",
    ]
    id_list = None
    start = 0
    max_results = 20
    for query in queries:
        papers = collect_arxiv_papers(
            search_query=query,
            id_list=id_list,
            start=start,
            max_results=max_results,
        )
        for paper in papers:
            paper_hash = generate_short_hash(paper.id)

            pdf_path = f"{paper_hash}.pdf"
            if not download_pdf(paper.pdf, pdf_path):
                continue

            images = get_pdf_page_images(pdf_path, dpi=150)
            images_predict = model.predict(images)
            images_usage = random.choices(
                ["train", "val", "test"],
                weights=[0.7, 0.15, 0.15],
                k=len(images),
            )

            for i in range(len(images)):
                image_id = f"{paper_hash}_{i}"

                images[i].save(f"dataset/images/{images_usage[i]}/{image_id}.png")
                results = images_predict[i]
                with open(f"dataset/labels/{images_usage[i]}/{image_id}.txt", "w") as f:
                    for result in results.boxes:
                        class_id = result.cls.item()
                        x_center, y_center, width, height = result.xywhn[0].tolist()
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

            os.remove(pdf_path)
            del images
            del images_predict
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
