import os

import pdfplumber
from nltk.tokenize import sent_tokenize

from text_preprocessing.named_entity_extract import get_named_entity_counts

if os.getcwd() == '/var/task':
    OUTPUT_DIR = '/tmp/output'
    DOWNLOAD_DIR = '/tmp/download'
    INPUT_DIR = '/tmp/input'
else:
    OUTPUT_DIR = './tmp/output'
    DOWNLOAD_DIR = './tmp/download'
    INPUT_DIR = './tmp/input'

# Create list to store docs from the data file
file_docs = []


def get_text_from_pdf(filename):
    """Function to extract the text from a PDF file using the pdfplumber package.
    Args:
        filename (str): filename of the pdf
    Returns:
        str: raw text string object of the data contained in the PDF file.
    """

    # Create str object to hold raw text
    my_str = str()

    # Extract text from PDF file by looping through different pages in the file
    with pdfplumber.open(filename) as pdf:
        for page in pdf.pages:
            my_str = my_str + ' ' + page.extract_text().lower()
            
    # Return the content
    return my_str


# Helper function to process (tokenize and turn into a string) .pdf files
def tokenize_pdf_files(pdf_filename):

    raw_text = get_text_from_pdf(pdf_filename)
    pdf_token = sent_tokenize(raw_text)

    return pdf_token, raw_text

# Helper function to process (tokenize and turn into a string) .txt files


def tokenize_txt_files(txt_filename):
    file_docs = list()
    # Get raw string of the .txt file
    with open(txt_filename) as f:
        raw_text = f.read()
        f.close()
    # Tokenize the sentences in the .txt file and save to list object
    with open(txt_filename) as f:
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
