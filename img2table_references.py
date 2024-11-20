from img2table.document import PDF

pdf = PDF("a4034881output.pdf",
          # pages=[0, 2],
          detect_rotation=False,
          pdf_text_extraction=True)

from img2table.ocr import TesseractOCR
from img2table.document import Image

# Instantiation of OCR
ocr = TesseractOCR(n_threads=1, lang="spa")

# Display extracted tables
from IPython.display import display_html, display

extracted_tables = pdf.extract_tables(ocr=ocr,
                                      implicit_rows=True,
                                      borderless_tables=True,
                                      min_confidence=50)
def get_cropped_table(table, page, pdf):
  bbox = table.bbox
  return Image.fromarray(pdf.images[page][bbox.y1:bbox.y2,bbox.x1:bbox.x2])

for page, tables in extracted_tables.items():
    for idx, table in enumerate(tables):
        display_html(table.html_repr(title=f"Page {page + 1} - Extracted table n°{idx + 1}"), raw=True)
        display(get_cropped_table(table, page, pdf))