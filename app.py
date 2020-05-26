import numpy as np
import nltk, gensim, glob, os
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
from datetime import date
from read_functions import extract_text_from_docx, extract_text_from_pdf, extract_text_from_doc
import io, sys, PyPDF2


# Get the current date for the filename
today = date.today()
today = today.strftime("%m-%d-%Y")

# Directory variables (root dir, data dir, etc.)
ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(os.path.abspath(os.curdir), 'data')

def get_docx_files():
    all_docx_files = [file for file in glob.glob(DATA_DIR + "\*.docx*")]
    return all_docx_files

def get_pdf_files():
    all_pdf_files = [file for file in glob.glob(DATA_DIR + "\*.pdf*")]
    return all_pdf_files

def get_doc_files():
    all_doc_files = [file for file in glob.glob(DATA_DIR + "\*.doc*")]
    return all_doc_files


# PDF files
def tokenize_pdf_files(pdf_filename):
    file_text_pdf = str()
    pdf_file_docs = []
    pdf_file_gen_obj = extract_text_from_pdf(pdf_filename)

    # Loop through pages in pdf generator object
    for page in pdf_file_gen_obj:
        file_text_pdf += ' ' + page
        file_text_pdf = file_text_pdf.lower()

    pdf_sent_tokenized = sent_tokenize(file_text_pdf)

    for line in pdf_sent_tokenized:
        pdf_file_docs.append(line)
        
    # Tokenize & process words in each sentence
    pdf_gen_docs = [gensim.utils.simple_preprocess(text, min_len=3) for text in pdf_file_docs]   

    return pdf_gen_docs
 


if __name__ == '__main__':
    print(sys.argv)
    
    if len(sys.argv) != 3:
        print('Incorrect number of parameters entered. Please refer to the README on the input parameters.')
        exit()
    # elif FileNotFoundError:
    #     print('Please check filenames and that docs are located in the ''data'' folder')
    #     print('One or both files were not found.')
    #     exit()
    else:
        doc_1 = sys.argv[1]
        doc_2 = sys.argv[2]
        
        # Get filenames and file extensions of the input docs
        doc_1_filename, doc_1_ext = os.path.splitext(sys.argv[1])[0], os.path.splitext(sys.argv[1])[1]
        doc_2_filename, doc_2_ext = os.path.splitext(sys.argv[2])[0], os.path.splitext(sys.argv[2])[1]
        
    file_docs = []
    
    # Determine file extensions
    # If .txt file:
    if doc_1_ext == '.txt':
        print('.txt file detected ...')
        with open(DATA_DIR + '\\' + doc_1) as f:
            tokens = sent_tokenize(f.read())
            for line in tokens:
                file_docs.append(line)

    # If .pdf file:
    elif doc_1_ext == '.pdf':
        pdfFileObj = open(DATA_DIR + '//' + doc_1, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        num_pages = pdfReader.numPages
        # Loop through pages and append to list
        for x in range(num_pages):
            pageObj = pdfReader.getPage(x)
            pdf_page = pageObj.extractText()
            file_docs.append(pdf_page)
    else:
        raise TypeError
        print('A non-txt and pdf file detected. Please use only .txt or .pdf files')
        exit()
    
    
    # Tokenize (and process) words for each sentence
    gen_docs = [gensim.utils.simple_preprocess(text, min_len=3) for text in file_docs]
    
    print(gen_docs)

    print("Number of documents (sentences) in first file:", len(gen_docs))

    # Create gensim dictionary with ID as the key, and word token as the value
    dictionary = gensim.corpora.Dictionary(gen_docs)
    # print(dictionary.token2id)

    ### Create a bag of words corpus, passing the tokenized list of words to the Dictionary.doc2bow()
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

    # TFIDF - down weights tokens (words) that appears frequently across documents
    # Words that occur more frequently across the documents get smaller weights
    tf_idf = gensim.models.TfidfModel(corpus)

    # Print token_id and the token frequency
    # for doc in tf_idf[corpus]:
    #     print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

    # Create index tempfile
    index_tmpfile = get_tmpfile("wordindex")

    # Create similarity measure object
    sims = gensim.similarities.Similarity(index_tmpfile,tf_idf[corpus],
                                            num_features=len(dictionary))

    # Create Query Document - how similar is this query document to each document in the index
    file2_docs = []

    # Determine file extensions
    # If .txt file:
    if doc_2_ext == '.txt':
        print('.txt file detected ...')
        with open(DATA_DIR + '\\' + doc_2) as f:
            tokens = sent_tokenize(f.read())
            for line in tokens:
                file2_docs.append(line)
                # tokenize & process words

    # If .pdf file:
    elif doc_2_ext == '.pdf':
        print('pdf file detected ...')
        pdfFileObj = open(DATA_DIR + '//' + doc_2, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        num_pages = pdfReader.numPages
        # Loop through pages and append to list
        for x in range(num_pages):
            pageObj = pdfReader.getPage(x)
            pdf_page = pageObj.extractText()
            file2_docs.append(pdf_page)
        
        # file2_docs = tokenize_pdf_files(DATA_DIR + '\\' + doc_2)
    else:
        raise TypeError('A non-txt and pdf file detected. Please use only .txt or .pdf files')
        exit()

    print("Number of documents (sentences) in second file:", len(file2_docs))
    

    # Array of averages (len = number of docs in the query)
    # Each entry in the list is the average similarity of the docs in the query doc compared to the corpus
    avg_sims = [] 
    
    # query_doc_bow = dictionary.doc2bow(query_doc)
    # # find similarity for each document
    # query_doc_tf_idf = tf_idf[query_doc_bow]
    # # print (document_number, document_similarity)
    # print('Comparing Result:', sims[query_doc_tf_idf]) 
    # # calculate sum of similarities for each query doc
    # sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
    # # calculate average of similarity for each query doc
    # avg = sum_of_sims / len(file_docs)
    # # print average of similarity for each query doc
    # print(f'avg: {sum_of_sims / len(file_docs)}')
    # # add average values into array
    # avg_sims.append(avg)  

    for line in file2_docs:
        # tokenize & process words
        query_doc = gensim.utils.simple_preprocess(line, min_len=3)
        # query_doc = gensim.utils.simple_preprocess(line, min_len=3)
        # create bag of words
        query_doc_bow = dictionary.doc2bow(query_doc)
        # find similarity for each document
        query_doc_tf_idf = tf_idf[query_doc_bow]
        # print (document_number, document_similarity)
        print('Comparing Result:', sims[query_doc_tf_idf]) 
        # calculate sum of similarities for each query doc
        sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
        # calculate average of similarity for each query doc
        avg = sum_of_sims / len(file_docs)
        # print average of similarity for each query doc
        print(f'avg: {sum_of_sims / len(file_docs)}')
        # add average values into array
        avg_sims.append(avg)  

    print(query_doc)
    print('AVERAGE SIMS: ' , avg_sims)    

    # calculate total average
    total_avg = ((np.sum(avg_sims, dtype=np.float)) / len(file2_docs))

    # round the value and multiply by 100 to format it as percentage
    percentage_of_similarity = round(float(total_avg) * 100)
    print('Similarity percentage: ' , percentage_of_similarity)

    # if percentage is greater than 100
    if percentage_of_similarity >= 100:
        percentage_of_similarity = 100


