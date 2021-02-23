import os

import tika
from nltk.tokenize import sent_tokenize

from text_preprocessing.named_entity_extract import get_named_entity_counts

# Directory variables (root dir, data dir, etc.)
ROOT_DIR = os.path.abspath(os.getcwd())
OUTPUT_DIR = os.path.abspath('output')
INPUT_DIR = os.path.abspath('input')

# Create list to store docs from the data file
file_docs = []


def get_text_from_pdf(filename):
    """Function to extract the text from a PDF file using the Tika package.
    Args:
        filename (str): filename of the pdf
    Returns:
        str: raw text string of the data contained in the PDF file.
    """

    # Initialize the tika VM
    tika.initVM()

    # Import the parser
    from tika import parser

    # Extract the text
    text = parser.from_file(filename)

    # Return the content

    return text['content']


# Helper function to process (tokenize and turn into a string) .pdf files
def tokenize_pdf_files(pdf_filename):

    raw_text = get_text_from_pdf(INPUT_DIR + os.sep + pdf_filename)
    pdf_token = sent_tokenize(raw_text)

    return pdf_token, raw_text

# Helper function to process (tokenize and turn into a string) .txt files
def tokenize_txt_files(txt_filename):
    file_docs = list()
    # Get raw string of the .txt file
    with open(INPUT_DIR + os.sep + txt_filename) as f:
        raw_text = f.read()
        f.close()
    # Tokenize the sentences in the .txt file and save to list object
    with open(INPUT_DIR + os.sep + txt_filename) as f:
        tokens = sent_tokenize(f.read())
        for line in tokens:
            file_docs.append(line)
        f.close()

    return file_docs, raw_text


def read_file(nlp, filename):

    print(f'Reading file ... {filename}')

    doc_ext = os.path.splitext(filename)[1]

    # If .txt file:
    if doc_ext == '.txt':
        file_docs, raw_text_string = tokenize_txt_files(filename)
        named_entities = get_named_entity_counts(nlp, raw_text_string)

    # If .pdf file:
    elif doc_ext == '.pdf':
        file2_docs, raw_text_string = tokenize_pdf_files(filename)
        named_entities = get_named_entity_counts(nlp, raw_text_string)

    # If other type of file ...
    else:
        raise TypeError(
            'A non-txt and pdf file detected. Please use only .txt or .pdf files')
        exit()

    return named_entities
