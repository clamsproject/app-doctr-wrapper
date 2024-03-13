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
        name="docTR Wrapper",
        description="End to end OCR for extracting text from timeframes",
        app_license="MIT",
        identifier="doctr-wrapper",
        url="https://github.com/clamsproject/app-easyocr-wrapper",
        analyzer_version='0.8.1',
        analyzer_license="Apache 2.0",
    )
    metadata.add_input(DocumentTypes.VideoDocument)
    metadata.add_input(AnnotationTypes.TimeFrame)
    metadata.add_output(DocumentTypes.TextDocument)
    metadata.add_output(Uri.SENTENCE)
    metadata.add_output(Uri.PARAGRAPH)
    metadata.add_output(Uri.TOKEN)
    metadata.add_output(AnnotationTypes.Alignment)
    metadata.add_output(AnnotationTypes.BoundingBox)

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
