import textract
from nltk.tokenize import sent_tokenize
import os
from text_preprocessing.named_entity_extract import get_named_entity_counts

# Directory variables (root dir, data dir, etc.)
ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(os.path.abspath(os.curdir), 'data')
QUERY_DIR = os.path.join(os.path.abspath(os.curdir), 'query')
OUTPUT_DIR = os.path.join(os.path.abspath(os.curdir), 'out')

# Create list to store docs from the data file  
file_docs = []


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

def read_file(nlp, filename):
    doc_filename, doc_ext, doc, doc_cleaned = get_clean_filename(filename)
        
    print(f'Reading file ... {doc_cleaned}')
    
    # If .txt file:
    if doc_ext == '.txt':
        # with open(DATA_DIR + '\\' + doc_2) as f:
        with open(doc) as f:
            tokens = sent_tokenize(f.read())
            for line in tokens:
                file_docs.append(line)
        # If .pdf file:
    elif doc_ext == '.pdf':
        file2_docs, pdf_str = tokenize_pdf_files(doc)
        pdf_entities = get_named_entity_counts(nlp, pdf_str)
    else:
        raise TypeError('A non-txt and pdf file detected. Please use only .txt or .pdf files')
        exit()
    
    return pdf_entities