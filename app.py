"""
wrapper for DocTR end to end OCR
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from math import floor, ceil
from typing import Tuple

import numpy as np
import torch
from clams import ClamsApp, Restifier
from doctr.models import ocr_predictor
from lapps.discriminators import Uri
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh


class DoctrWrapper(ClamsApp):

    def __init__(self):
        super().__init__()
        # default docTR configs: 
        # det_arch='db_resnet50' (keeping it)
        # reco_arch='crnn_vgg16_bn', 
        # pretrained=False, 
        # paragraph_break=0.035, (keeping it)
        # assume_straight_pages=True
        # detect_orientation=False, 
        self.reader = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq',
                                    pretrained=True, 
                                    paragraph_break=0.035,
                                    assume_straight_pages=False, detect_orientation=True)
        if torch.cuda.is_available():
            self.gpu = True
            self.reader = self.reader.cuda().half()
        else:
            self.gpu = False

    def _appmetadata(self):
        # using metadata.py
        pass

    @staticmethod
    def rel_coords_to_abs(coords: Tuple[Tuple[float, float]], width: int, height: int) -> Tuple[Tuple[int, int]]:
        """
        Simple conversion from relative coordinates (percentage) to absolute coordinates (pixel). 
        Assumes the passed shape is a rectangle, represented by top-left and bottom-right corners, 
        and compute floor and ceiling based on the geometry.
        """
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        return (floor(x1 * height), floor(y1 * width)), (ceil(x2 * height), ceil(y2 * width))
    
    @staticmethod
    def create_bbox(new_view: View, 
                    coordinates: Tuple[Tuple[int, int]],
                    timepoint_ann: Annotation, text_ann: Annotation):
        bbox_ann = new_view.new_annotation(AnnotationTypes.BoundingBox, coordinates=coordinates, label="text")
        new_view.new_annotation(AnnotationTypes.Alignment, source=timepoint_ann.id, target=bbox_ann.id)
        new_view.new_annotation(AnnotationTypes.Alignment, source=text_ann.id, target=bbox_ann.id)

    def process_timepoint(self, representative: Annotation, new_view: View, video_doc: Document):
        rep_frame_index = vdh.convert(representative.get("timePoint"),
                                      representative.get("timeUnit"), "frame", 
                                      video_doc.get("fps"))
        image: np.ndarray = vdh.extract_frames_as_images(video_doc, [rep_frame_index], as_PIL=False)[0]
        h, w = image.shape[:2]
        result = self.reader([image])
        # assume only one page, as we are passing one image at a time
        text_content = result.render()
        if not text_content:
            return representative.get('timePoint'), None
        text_document: Document = new_view.new_textdocument(result.render())
        td_id = text_document.id
        new_view.new_annotation(AnnotationTypes.Alignment, source=representative.id, target=td_id)

        e = 0
        for block in result.pages[0].blocks:
            para_ann = new_view.new_annotation(Uri.PARAGRAPH, document=td_id, text=block.render())
            self.create_bbox(new_view, self.rel_coords_to_abs(block.geometry, w, h), representative, para_ann)
            target_sents = []
            
            for line in block.lines:
                sent_ann = new_view.new_annotation(Uri.SENTENCE, document=td_id, text=line.render())
                target_sents.append(sent_ann.id)
                self.create_bbox(new_view, self.rel_coords_to_abs(line.geometry, w, h), representative, sent_ann)
                target_tokens = []
                
                for word in line.words:
                    s = text_content.find(word.value, e)
                    e = s + len(word.value)
                    token_ann = new_view.new_annotation(Uri.TOKEN, document=td_id, start=s, end=e, text=word.value)
                    target_tokens.append(token_ann.id)
                    self.create_bbox(new_view, self.rel_coords_to_abs(word.geometry, w, h), representative, token_ann)
                sent_ann.add_property("targets", target_tokens)
            para_ann.add_property("targets", target_sents)

        return representative.get('timePoint'), text_content

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        if self.gpu:
            self.logger.debug("running app on GPU")
        else:
            self.logger.debug("running app on CPU")
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view: View = mmif.get_views_for_document(video_doc.properties.id)[-1]

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.BoundingBox)
        new_view.new_contain(AnnotationTypes.Alignment)
        new_view.new_contain(Uri.PARAGRAPH)
        new_view.new_contain(Uri.SENTENCE)
        new_view.new_contain(Uri.TOKEN)

        with ThreadPoolExecutor() as executor:
            futures = []
            for timeframe in input_view.get_annotations(AnnotationTypes.TimeFrame):
                for rep_id in timeframe.get("representatives"):
                    if Mmif.id_delimiter not in rep_id:
                        rep_id = f'{input_view.id}{Mmif.id_delimiter}{rep_id}'
                    representative = mmif[rep_id]
                    futures.append(executor.submit(self.process_timepoint, representative, new_view, video_doc))
                if len(futures) == 0:
                    # TODO (krim @ 4/18/24): if "representatives" is not present, process just the middle frame
                    pass

            for future in futures:
                timestemp, text_content = future.result()
                self.logger.debug(f'Processed timepoint: {timestemp}, recognized text: "{json.dumps(text_content)}"')

        return mmif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = DoctrWrapper()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
