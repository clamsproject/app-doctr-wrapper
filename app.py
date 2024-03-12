"""
wrapper for DocTR end to end OCR
"""

import argparse
import logging
from typing import Union

from lapps.discriminators import Uri

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

from doctr.models import ocr_predictor
import torch
import numpy as np


def create_bbox(view: View, coordinates, box_type, time_point):
    bbox = view.new_annotation(AnnotationTypes.BoundingBox)
    bbox.add_property("coordinates", coordinates)
    bbox.add_property("boxType", box_type)
    bbox.add_property("timePoint", time_point)
    return bbox


def create_alignment(view: View, source, target) -> None:
    alignment = view.new_annotation(AnnotationTypes.Alignment)
    alignment.add_property("source", source)
    alignment.add_property("target", target)


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

    class Paragraph:
        """
        lapps annotation corresponding to a DocTR Block object targeting contained sentences.
        """
        def __init__(self, region: Annotation, document: Document):
            self.region = region
            self.region.add_property("document", document.id)
            self.sentences = []

        def add_sentence(self, sentence):
            self.sentences.append(sentence)

        def collect_targets(self):
            self.region.add_property("targets", [s.region.id for s in self.sentences])

    class Sentence:
        """
        Span annotation corresponding to a DocTR Line object targeting contained tokens.
        """
        def __init__(self, region: Annotation, document: Document):
            self.region = region
            self.region.add_property("document", document.id)
            self.tokens = []

        def add_token(self, token):
            self.tokens.append(token)

        def collect_targets(self):
            self.region.add_property("targets", [t.region.id for t in self.tokens])

    class Token:
        """
        Span annotation corresponding to a DocTR Word object. Start and end are character offsets in the text document.
        """
        def __init__(self, region: Annotation, document: Document, start: int, end: int):
            self.region = region
            self.region.add_property("document", document.id)
            self.region.add_property("start", start)
            self.region.add_property("end", end)

    def process_timeframe(self, timeframe: Annotation, new_view: View, video_doc: Document, input_view: View):
        representative: AnnotationTypes.TimePoint = input_view.get_annotation_by_id(timeframe.get("representatives")[0])
        rep_frame_index = vdh.convert(representative.get("timePoint"), "milliseconds",
                                      "frame", vdh.get_framerate(video_doc))
        image: np.ndarray = vdh.extract_frames_as_images(video_doc, [rep_frame_index], as_PIL=False)[0]
        extracted_text = ""
        result = self.reader([image])
        blocks = result.pages[0].blocks
        text_document = new_view.new_textdocument(result.render())

        for block in blocks:
            try:
                extracted_text = self.process_block(block, new_view, text_document, representative, extracted_text)
            except Exception as e:
                self.logger.error(f"Error processing block: {e}")
                continue

        return extracted_text, text_document

    def process_block(self, block, view, text_document, representative, extracted_text):
        paragraph = self.Paragraph(view.new_annotation(at_type=Uri.PARAGRAPH), text_document)
        paragraph_bb = create_bbox(view, block.geometry, "text", representative.id)
        create_alignment(view, paragraph.region.id, paragraph_bb.id)

        for line in block.lines:
            try:
                sentence, extracted_text = self.process_line(line, view, text_document, representative, extracted_text)
            except Exception as e:
                self.logger.error(f"Error processing line: {e}")
                continue
            paragraph.add_sentence(sentence)

        paragraph.collect_targets()
        return extracted_text

    def process_line(self, line, view, text_document, representative, extracted_text):
        sentence = self.Sentence(view.new_annotation(at_type=Uri.SENTENCE), text_document)
        sentence_bb = create_bbox(view, line.geometry, "text", representative.id)
        create_alignment(view, sentence.region.id, sentence_bb.id)

        for word in line.words:
            if word.confidence > 0.4:
                start = text_document.text_value.find(word.value)
                end = start + len(word.value)
                token = self.Token(view.new_annotation(at_type=Uri.TOKEN), text_document, start, end)
                token_bb = create_bbox(view, word.geometry, "text", representative.id)
                create_alignment(view, token.region.id, token_bb.id)
                sentence.add_token(token)
                extracted_text += word.value + " "

        sentence.collect_targets()
        return sentence, extracted_text

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        self.logger.debug("running app")
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view: View = mmif.get_views_for_document(video_doc.properties.id)[0]

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)

        for timeframe in input_view.get_annotations(AnnotationTypes.TimeFrame):
            try:
                extracted_text, text_document = self.process_timeframe(timeframe, new_view, video_doc, input_view)
                self.logger.debug(extracted_text)
                self.logger.debug(text_document.get('text'))
                representative: Annotation = input_view.get_annotation_by_id(timeframe.get("representatives")[0])
                create_alignment(new_view, representative.id, text_document.id)
            except Exception as e:
                self.logger.error(f"Error processing timeframe: {e}")
                continue

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
