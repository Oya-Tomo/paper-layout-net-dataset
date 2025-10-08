import os
import random
from dataclasses import dataclass

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


@dataclass
class DatasetConfig:
    dpi: int = 150
    batch_size: int = 15
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


def main(config: DatasetConfig = DatasetConfig()):
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

    page_pool: list[
        tuple[
            str,
            int,
            Image.Image,
            str,
        ]
    ] = []  # contains (hash, page, image, usage)
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

            images = get_pdf_page_images(pdf_path, dpi=config.dpi)
            images_usage = random.choices(
                ["train", "val", "test"],
                weights=[config.train_split, config.val_split, config.test_split],
                k=len(images),
            )
            for idx, image in enumerate(images):
                page_pool.append((paper_hash, idx, image, images_usage[idx]))

            os.remove(pdf_path)

            if len(page_pool) < config.batch_size:
                continue

            for _ in range(len(page_pool) // config.batch_size):
                batch = page_pool[: config.batch_size]
                page_pool = page_pool[config.batch_size :]

                paper_hashes, page_indices, images, images_usage = zip(*batch)
                images = list(images)
                images_usage = list(images_usage)
                images_predict = model.predict(images)

                for i in range(len(images)):
                    image_id = f"{paper_hashes[i]}_{page_indices[i]}"

                    images[i].save(f"dataset/images/{images_usage[i]}/{image_id}.png")
                    results = images_predict[i]
                    with open(
                        f"dataset/labels/{images_usage[i]}/{image_id}.txt", "w"
                    ) as f:
                        for result in results.boxes:
                            class_id = int(result.cls.item())
                            x_center, y_center, width, height = result.xywhn[0].tolist()
                            f.write(
                                f"{class_id} {x_center} {y_center} {width} {height}\n"
                            )

                del images_predict
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
