import docx2txt
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io, glob
import textract, spacy, re, os
from nltk import word_tokenize, sent_tokenize

# Directory variables (root dir, data dir, etc.)
ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(os.path.abspath(os.curdir), 'data')

# Function used to extract data from .docx files
def extract_text_from_docx(doc_path):
    temp = docx2txt.process(doc_path)
    text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
    return ' '.join(text)

# Function used to extract data from .pdf files
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as fh:
        # iterate over all pages of PDF document
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            # creating a resoure manager
            resource_manager = PDFResourceManager()
            
            # create a file handle
            fake_file_handle = io.StringIO()
            
            # creating a text converter object
            converter = TextConverter(
                                resource_manager, 
                                fake_file_handle, 
                                codec='utf-8', 
                                laparams=LAParams()
                        )

            # creating a page interpreter
            page_interpreter = PDFPageInterpreter(
                                resource_manager, 
                                converter
                            )

            # process current page
            page_interpreter.process_page(page)
            
            # extract text
            text = fake_file_handle.getvalue()
            yield text

            # close open handles
            converter.close()
            fake_file_handle.close()

def extract_text_from_doc(doc_filename):
    str_text_doc = textract.process(doc_filename)
    file_doc = []
    revised_list = []
    
    str_text_doc_string = str(str_text_doc)
    
    str_text_doc_string = str_text_doc_string.replace('\\r',' ').replace('\\n',' ').replace('\\', '').replace('  ', ' ').replace("\'", "").strip()
    
    file_doc = str_text_doc_string.split(sep=".")
    
    for line in file_doc:
        revised_list.append(re.sub(' +', ' ', line))
    
    output_string = str(revised_list)
    
    # nlp = spacy.load("en_core_web_sm")
    # nlp_text = nlp(str_text_doc_string)
    
    # tokens = [token.text for token in nlp_text if not token.is_stop]
    
    # for token in tokens:
    #     if token.lower() in skills:
    #         skillset.append(token)
    
    return output_string

def tokenize_doc_files(doc_filename):
    file_text_doc = str()
    doc_file_docs = []
    doc_file = extract_text_from_doc(doc_filename)
    
    doc_sent_tokenized = sent_tokenize(doc_file)  
    
    for line in doc_sent_tokenized:
        doc_file_docs.append(line)
        
    # Tokenize words for each sentence
    doc_gen_docs = [[w.lower() for w in word_tokenize(text)] for text in doc_file_docs]   
        
    return doc_gen_docs   
    
# Function to get list of documents in a provided directory
def get_docs(location= os.curdir):
    dir_files = [file for file in glob.glob(DATA_DIR + "\*.txt*")]
    return dir_files

# DOCX files
def tokenize_docx_files(docx_filename):
    docx_file_docs = []
    docx_str = extract_text_from_docx(DATA_DIR + docx_filename)
    docx_sent_tokenized = sent_tokenize(docx_str)
    for line in docx_sent_tokenized:
        docx_file_docs.append(line)

    # Tokenize words for each sentence
    docx_gen_docs = [[w.lower() for w in word_tokenize(text)] for text in docx_file_docs]
    
    return docx_gen_docs