from ultralytics.data.utils import visualize_image_annotations

label_map = (
    {
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
)

visualize_image_annotations(
    label_map=label_map,
)
