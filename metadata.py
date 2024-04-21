"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes
from lapps.discriminators import Uri

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    metadata = AppMetadata(
        name="CLAMS docTR Wrapper",
        description='CLAMS app wraps the docTR, End-to-End OCR model, available at '
                    'https://pypi.org/project/python-doctr . The model is capable of detecting text regions in the '
                    'input image and recognizing text in the regions. The text-localized regions are organized '
                    'hierarchically by the model into "pages" > "blocks" > "lines" > "words", and this CLAMS app '
                    'translated into `TextDocument`, `Paragraphs`, `Sentence`, and `Token` annotations that represent '
                    'recognized text contents, then aligned to `BoundingBox` annotations that represent the detected '
                    'geometries.',
        app_license="Apache 2.0",
        identifier="doctr-wrapper",
        url="https://github.com/clamsproject/app-doctr-wrapper",
        analyzer_version='0.8.1',
        analyzer_license="Apache 2.0",
    )
    metadata.add_input(DocumentTypes.VideoDocument)
    in_tf = metadata.add_input(AnnotationTypes.TimeFrame, representatives='?')
    in_tf.add_description('The Time frame annotation that represents the video segment to be processed. When '
                          '`representatives` property is present, the app will process videos still frames at the '
                          'underlying time point annotations that are referred to by the `representatives` property. '
                          'Otherwise, the app will process the middle frame of the video segment.')
    metadata.add_output(DocumentTypes.TextDocument)
    out_sent = metadata.add_output(at_type=Uri.SENTENCE)
    out_sent.add_description('Translation of the recognized "text lines" in the processed input images')
    out_para = metadata.add_output(at_type=Uri.PARAGRAPH)
    out_para.add_description('Translation of the recognized "text blocks" in the processed input images')
    out_tkn = metadata.add_output(at_type=Uri.TOKEN)
    out_tkn.add_description('Translation of the recognized "text words" in the processed input images')
    out_ali = metadata.add_output(AnnotationTypes.Alignment)
    out_ali.add_description('Alignments between 1) `TimePoint` <-> `TextDocument`, 2) `TimePoint` <-> '
                            '`Token`/`Sentence`/`Paragraph`, 3) `BoundingBox` <-> `Token`/`Sentence`/`Paragraph`')
    out_bbox = metadata.add_output(AnnotationTypes.BoundingBox)
    out_bbox.add_description('Bounding boxes of the detected text regions in the input images. No corresponding box '
                             'for the entire image (`TextDocument`) region')
    
    metadata.add_parameter(name='tfLabel', default=[], type='string', multivalued=True,
                           description='The label of the TimeFrame annotation to be processed. By default (`[]`), all '
                                       'TimeFrame annotations will be processed, regardless of their `label` property '
                                       'values.')

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
