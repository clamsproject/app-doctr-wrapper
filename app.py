"""
wrapper for DocTR end to end OCR
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from clams import ClamsApp, Restifier
from doctr.models import ocr_predictor
from lapps.discriminators import Uri
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh


# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes


def create_bbox(view: View, coordinates, box_type, time_point):
    bbox = view.new_annotation(AnnotationTypes.BoundingBox)
    bbox.add_property("coordinates", coordinates)
    bbox.add_property("label", box_type)
    bbox.add_property("timePoint", time_point)
    return bbox


def create_alignment(view: View, source, target) -> None:
    alignment = view.new_annotation(AnnotationTypes.Alignment)
    alignment.add_property("source", source)
    alignment.add_property("target", target)


class DoctrWrapper(ClamsApp):

    def __init__(self):
        super().__init__()
        self.reader = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq',
                                    pretrained=True, detect_orientation=True, paragraph_break=0.035,
                                    assume_straight_pages=True)
        if torch.cuda.is_available():
            self.gpu = True
            self.reader = self.reader.cuda().half()
        else:
            self.gpu = False

    def _appmetadata(self):
        # using metadata.py
        pass
    
    class LingUnit(object):
        """
        A thin wrapper for LAPPS linguistic unit annotations that 
        represent different geometric levels from DocTR OCR output.
        """
        def __init__(self, region: Annotation, document: Document):
            self.region = region
            self.region.add_property("document", document.id)
            self.children = []

        def add_child(self, sentence):
            self.children.append(sentence)

        def collect_targets(self):
            self.region.add_property("targets", [child.region.id for child in self.children])

    class Token:
        """
        Span annotation corresponding to a DocTR Word object. Start and end are character offsets in the text document.
        """
        def __init__(self, region: Annotation, document: Document, start: int, end: int):
            self.region = region
            self.region.add_property("document", document.id)
            self.region.add_property("start", start)
            self.region.add_property("end", end)

    def process_timepoint(self, representative: Annotation, new_view: View, video_doc: Document):
        rep_frame_index = vdh.convert(representative.get("timePoint"), 
                                      representative.get("timeUnit"), "frame", 
                                      video_doc.get("fps"))
        image: np.ndarray = vdh.extract_frames_as_images(video_doc, [rep_frame_index], as_PIL=False)[0]
        result = self.reader([image])
        blocks = result.pages[0].blocks
        text_document: Document = new_view.new_textdocument(result.render())

        for block in blocks:
            try:
                self.process_block(block, new_view, text_document, representative)
            except Exception as e:
                self.logger.error(f"Error processing block: {e}")
                continue

        return text_document, representative

    def process_block(self, block, view, text_document, representative):
        paragraph = self.LingUnit(view.new_annotation(at_type=Uri.PARAGRAPH), text_document)
        paragraph_bb = create_bbox(view, block.geometry, "text", representative.id)
        create_alignment(view, paragraph.region.id, paragraph_bb.id)

        for line in block.lines:
            try:
                sentence = self.process_line(line, view, text_document, representative)
            except Exception as e:
                self.logger.error(f"Error processing line: {e}")
                continue
            paragraph.add_child(sentence)
        paragraph.collect_targets()

    def process_line(self, line, view, text_document, representative):
        sentence = self.LingUnit(view.new_annotation(at_type=Uri.SENTENCE), text_document)
        sentence_bb = create_bbox(view, line.geometry, "text", representative.id)
        create_alignment(view, sentence.region.id, sentence_bb.id)

        for word in line.words:
            if word.confidence > 0.4:
                start = text_document.text_value.find(word.value)
                end = start + len(word.value)
                token = self.Token(view.new_annotation(at_type=Uri.TOKEN), text_document, start, end)
                token_bb = create_bbox(view, word.geometry, "text", representative.id)
                create_alignment(view, token.region.id, token_bb.id)
                sentence.add_child(token)

        sentence.collect_targets()
        return sentence

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        if self.gpu:
            self.logger.debug("running app on GPU")
        else:
            self.logger.debug("running app on CPU")
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view: View = mmif.get_views_for_document(video_doc.properties.id)[-1]

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)

        with ThreadPoolExecutor() as executor:
            futures = []
            for timeframe in input_view.get_annotations(AnnotationTypes.TimeFrame):
                for rep_id in timeframe.get("representatives"):
                    if Mmif.id_delimiter not in rep_id:
                        rep_id = f'{input_view.id}{Mmif.id_delimiter}{rep_id}'
                    representative = mmif[rep_id]
                    futures.append(executor.submit(self.process_timepoint, representative, new_view, video_doc))

            for future in futures:
                try:
                    text_document, representative = future.result()
                    self.logger.debug(text_document.get('text'))
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
