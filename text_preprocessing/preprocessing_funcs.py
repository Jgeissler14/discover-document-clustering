import textract
from nltk.tokenize import sent_tokenize
import os

# Directory variables (root dir, data dir, etc.)
ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(os.path.abspath(os.curdir), 'data')
QUERY_DIR = os.path.join(os.path.abspath(os.curdir), 'query')
OUTPUT_DIR = os.path.join(os.path.abspath(os.curdir), 'out')

# Read in PDF file and return list of unprocessed docs (sentences) from the file
def tokenize_pdf_files(pdf_filename):
    raw_text = textract.process(pdf_filename, encoding='utf-8')
    str_raw_text = raw_text.decode('utf-8')
    pdf_token = sent_tokenize(str_raw_text)
    
    return pdf_token, str_raw_text

def get_clean_filename(filename):
    # Extract file info (extension, filename, filename without extension)
    doc_filename, doc_ext = os.path.splitext(filename)[0], os.path.splitext(filename)[1]
    doc = doc_filename + doc_ext
    doc_cleaned =  doc.replace(QUERY_DIR + '\\', '')
    doc_cleaned = doc_cleaned.replace(DATA_DIR + '\\', '')
    
    return doc_filename, doc_ext, doc, doc_cleaned