import os

import textract
from nltk.tokenize import sent_tokenize

from text_preprocessing.named_entity_extract import get_named_entity_counts

# Directory variables (root dir, data dir, etc.)
ROOT_DIR = os.path.abspath(os.getcwd())
OUTPUT_DIR = os.path.abspath('output')
INPUT_DIR = os.path.abspath('input')
# QUERY_DIR = os.path.abspath('../query')

# Create list to store docs from the data file
file_docs = []


# Read in PDF file and return list of unprocessed docs (sentences) from the file
def tokenize_pdf_files(pdf_filename):
    raw_text = textract.process(
        INPUT_DIR + os.sep + pdf_filename, encoding='utf-8')
    str_raw_text = raw_text.decode('utf-8')
    pdf_token = sent_tokenize(str_raw_text)

    return pdf_token, str_raw_text


def tokenize_txt_files(txt_filename):
    file_docs = list()
    with open(INPUT_DIR + os.sep + txt_filename) as f:
        tokens = sent_tokenize(f.read())
        for line in tokens:
            file_docs.append(line)
    return file_docs


def read_file(nlp, filename):

    print(f'Reading file ... {filename}')

    doc_ext = os.path.splitext(filename)[1]

    # If .txt file:
    if doc_ext == '.txt':
        file_docs = tokenize_txt_files(filename)

    # If .pdf file:
    elif doc_ext == '.pdf':
        file2_docs, pdf_str = tokenize_pdf_files(filename)
        pdf_entities = get_named_entity_counts(nlp, pdf_str)
    
    # If other type of file ...
    else:
        raise TypeError(
            'A non-txt and pdf file detected. Please use only .txt or .pdf files')
        exit()

    return pdf_entities
