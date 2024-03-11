# docTR-wrapper

## Description

Wrapper for [docTR](https://github.com/mindee/doctr) end-to-end text detection and recognition.

## Input

The wrapper takes a [`VideoDocument`]('https://mmif.clams.ai/vocabulary/VideoDocument/v1/') with SWT 
[`TimeFrame`]('https://mmif.clams.ai/vocabulary/TimeFrame/v3/') annotations. The app specifically 
uses the representative `TimePoint` annotations from SWT v4 `TimeFrame` annotations to extract specific frames for OCR

## docTR Structured Output

[From the docTR documentation]('https://mindee.github.io/doctr/latest//using_doctr/using_models.html')

The docTR model returns a [`Document` object]('https://mindee.github.io/doctr/latest//modules/io.html#document-structure')

Here is the typical *Document* layout:
```
Document(
  (pages): [Page(
    dimensions=(340, 600)
    (blocks): [Block(
      (lines): [Line(
        (words): [
          Word(value='No.', confidence=0.91),
          Word(value='RECEIPT', confidence=0.99),
          Word(value='DATE', confidence=0.96),
        ]
      )]
      (artefacts): []
    )]
  )]
)
```

The docTR wrapper preserves this structured information in the output MMIF by creating 
lapps `Paragraph` `Sentence` and `Token` annotations corresponding to the `Block`, `Line`, and `Word` from the docTR output.

## User instruction

General user instructions for CLAMS apps is available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### System requirements

- Requires mmif-python[cv] for the `VideoDocument` helper functions
- Requires GPU to run at a reasonable speed

### Configurable runtime parameter

For the full list of parameters, please refer to the app metadata from [CLAMS App Directory](https://apps.clams.ai) or [`metadata.py`](metadata.py) file in this repository.