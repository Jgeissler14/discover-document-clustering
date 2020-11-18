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
    raw_text = textract.process(INPUT_DIR + os.sep + pdf_filename, encoding='utf-8')
    str_raw_text = raw_text.decode('utf-8')
    pdf_token = sent_tokenize(str_raw_text)

    return pdf_token, str_raw_text


# def get_clean_filename(filename):
#     # print(f'in get_clean_filename for {filename}')
#     # Extract file info (extension, filename, filename without extension)
#     doc_filename, doc_ext = os.path.splitext(filename)[0], os.path.splitext(filename)[1]
#     doc = doc_filename + doc_ext
#     doc_cleaned = doc.replace(QUERY_DIR + '\\', '')
#     doc_cleaned = doc_cleaned.replace(INPUT_DIR + '\\', '')

#     # print(doc_filename, doc_ext, doc, doc_cleaned)

#     return doc_filename, doc_ext, doc, doc_cleaned


def read_file(nlp, filename):
    # doc_filename, doc_ext, doc, doc_cleaned = get_clean_filename(filename)
    
    # print(f'doc_filename: {doc_filename}')
    # print(f'doc_ext: {doc_ext}')
    # print(f'doc: {doc}')
    # print(f'doc_cleaned: {doc_cleaned}')

    print(f'Reading file ... {filename}')
    
    doc_ext = os.path.splitext(filename)[1]

    # If .txt file:
    if doc_ext == '.txt':
        # with open(INPUT_DIR + '\\' + doc_2) as f:
        with open(filename) as f:
            tokens = sent_tokenize(f.read())
            for line in tokens:
                file_docs.append(line)
        # If .pdf file:
    elif doc_ext == '.pdf':
        file2_docs, pdf_str = tokenize_pdf_files(filename)
        pdf_entities = get_named_entity_counts(nlp, pdf_str)
    else:
        raise TypeError('A non-txt and pdf file detected. Please use only .txt or .pdf files')
        exit()

    return pdf_entities
