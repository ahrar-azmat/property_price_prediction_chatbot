import os
import docx
import PyPDF2
import xlrd
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_file(file_path):
    """
    Extract text from various file types including txt, docx, pdf, xls, xlsx, and csv.

    :param file_path: Path to the file
    :return: Extracted text as a string
    """
    extension = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if extension == ".txt":
            with open(file_path, 'r') as file:
                text = file.read()
        elif extension == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif extension == ".pdf":
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfFileReader(file)
                for page in range(reader.numPages):
                    text += reader.getPage(page).extract_text()
        elif extension in [".xls", ".xlsx"]:
            workbook = xlrd.open_workbook(file_path)
            sheet = workbook.sheet_by_index(0)
            for row in range(sheet.nrows):
                text += " ".join([str(cell.value) for cell in sheet.row(row)]) + "\n"
        elif extension == ".csv":
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
        logger.debug(f"Extracted text from {file_path}")
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
    return text
