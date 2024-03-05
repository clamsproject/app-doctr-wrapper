"""
wrapper for DocTR end to end OCR
"""

import argparse
import logging
from typing import Union

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

from doctr.models import ocr_predictor
import torch
import numpy as np


class DoctrWrapper(ClamsApp):

    def __init__(self):
        super().__init__()
        self.reader = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_mobilenet_v3_large',
                                    pretrained=True, detect_orientation=True, paragraph_break=0.015,
                                    assume_straight_pages=True).to(torch.device("cuda:0"))

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        self.logger.debug("running app")
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view: View = mmif.get_views_for_document(video_doc.properties.id)[0]

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)

        for timeframe in input_view.get_annotations(AnnotationTypes.TimeFrame):
            self.logger.debug(timeframe.properties)
            representative: AnnotationTypes.TimePoint = (
                input_view.get_annotation_by_id(timeframe.get("representatives")[0]))
            self.logger.debug("Sampling 1 frame")
            rep_frame = vdh.convert(representative.get("timePont"), "milliseconds",
                                    "frame", vdh.get_framerate(video_doc))
            image: np.ndarray = vdh.extract_frames_as_images(video_doc, [rep_frame], as_PIL=False)[0]
            self.logger.debug("Extracted image")
            self.logger.debug("Running OCR")
            ocrs = []
            result = self.reader([image])
            blocks = result.pages[0].blocks
            for block in blocks:
                for line in block.lines:
                    for word in line.words:
                        ocrs.append((word.geometry, word.value, word.confidence))
            self.logger.debug(ocrs)
            timepoint = representative
            for ocr in ocrs:
                for coord, text, score in ocr:
                    if score > 0.4:
                        self.logger.debug("Confident OCR: " + text)
                        text_document = new_view.new_textdocument(text)
                        bbox_annotation = new_view.new_annotation(AnnotationTypes.BoundingBox)
                        bbox_annotation.add_property("coordinates", coord)
                        bbox_annotation.add_property("boxType", "text")
                        time_bbox = new_view.new_annotation(AnnotationTypes.Alignment)
                        time_bbox.add_property("source", timepoint.id)
                        time_bbox.add_property("target", bbox_annotation.id)
                        bbox_text = new_view.new_annotation(AnnotationTypes.Alignment)
                        bbox_text.add_property("source", bbox_annotation.id)
                        bbox_text.add_property("target", text_document.id)

        return mmif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen" )
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
