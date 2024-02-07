import doctr.io
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import torch
import numpy as np
import matplotlib.pyplot as plt
from doctr.utils.visualization import visualize_page
import argparse
from PIL import Image
import json
import cv2 as cv


class OCR:
    def __init__(self, image):
        self.reader = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_mobilenet_v3_large', pretrained=True, detect_orientation=True, paragraph_break=0.015, assume_straight_pages=True).to(torch.device("cuda:0"))
        self.image = image
        self.results = self.reader(image)


def draw(results: doctr.io.Document, image):
    print('drawing')
    annotated_img = image.copy()
    height, width = results.pages[0].dimensions
    print(results.pages[0].blocks)
    for block in results.pages[0].blocks:
        block_bb = ((int(block.geometry[0][0] * width), int(block.geometry[0][1] * height)),
                    (int(block.geometry[1][0] * width), int(block.geometry[1][1] * height)))
        print(f'drawing block with {block_bb}')
        annotated_img = cv.rectangle(annotated_img, block_bb[0], block_bb[1], (255, 0, 0), 3)
        for line in block.lines:
            line_bb = ((int(line.geometry[0][0] * width ), int(line.geometry[0][1] * height)),
                        (int(line.geometry[1][0] * width), int(line.geometry[1][1] * height)))
            print(f'drawing line with {line_bb}')
            annotated_img = cv.rectangle(annotated_img, line_bb[0], line_bb[1], (0, 255, 0), 2)
    return annotated_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Path to image')
    args = parser.parse_args()
    image = DocumentFile.from_images(args.image)
    document = np.asarray(Image.open(args.image))
    ocr = OCR(image)
    with open('ocr_results.json', 'w') as file:
        json.dump(ocr.results.export(), file)
    cv.imwrite('ocr_draw.jpg', draw(ocr.results, document))
    ocr_viz = visualize_page(ocr.results.pages[0].export(), document, words_only=False)
    plt.savefig("ocr_visualizer.png")

