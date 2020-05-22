import docx2txt
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
import textract, spacy, re

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
    