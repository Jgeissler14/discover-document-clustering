import numpy as np
import gensim, glob, os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk_sents, defaultdict
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
from datetime import datetime
import io, sys, textract
import pandas as pd
import spacy, re
from collections import Counter


# Get the current date for the filename
today = datetime.today()
today_str = today.strftime("%m-%d-%Y-%H%M%S")

# Directory variables (root dir, data dir, etc.)
ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(os.path.abspath(os.curdir), 'data')
QUERY_DIR = os.path.join(os.path.abspath(os.curdir), 'query')
OUTPUT_DIR = os.path.join(os.path.abspath(os.curdir), 'out')

# List of supported file extensions
supported_files = ["\*.pdf*", "\*.txt*"] 

# List to gather all filenames in the 'data' directory
all_files = list()
query_files = list()

# Find all valid files in the 'data' directory and append to 'all_files' list
for extension in supported_files:   
    for file in glob.glob(DATA_DIR + extension):
        all_files.append(file)

# Find all valid files in the 'query' directory and append to 'all_files_query' list
for extension in supported_files:   
    for file in glob.glob(QUERY_DIR + extension):
        query_files.append(file)

# Initialize df to store comparison results   
results_df = pd.DataFrame(columns=['file_1','file_2','total_avg','similarity_rating'])

# Read in PDF file and return list of unprocessed docs (sentences) from the file
def tokenize_pdf_files(pdf_filename):
    # raw_text = textract.process(DATA_DIR + '//' + pdf_filename, encoding='utf-8')
    raw_text = textract.process(pdf_filename, encoding='utf-8')
    str_raw_text = raw_text.decode('utf-8')
    pdf_token = sent_tokenize(str_raw_text)
    
    return pdf_token, str_raw_text
 
def named_entity_counts(raw_text):
    # Create spacy doc
    nlp = spacy.load('en_core_web_sm')
    # nlp = spacy.load('en_core_web_lg')
    doc = nlp(raw_text)
    
    # Store the number of words in the doc
    tot_num_words = len(doc)

    # Create list of named entities along with their labels
    doc_entities = [[re.sub('  ', ' ', re.sub('\r',' ',re.sub('\n', ' ', ent.text))).strip(), ent.label_, ] 
                    for ent in doc.ents if ent.text.isnumeric() == False and ent.text.startswith('https://') == False] 
    

    # Create defaultdict to list the entities and their counts
    entities_cnt = defaultdict(int)
    for entity, ent_label in doc_entities:
        entities_cnt[entity] +=1

    # Sort the entities list by the number of occurrences 
    entities_cnt_sorted = sorted(entities_cnt.items(), key=lambda x: x[1], reverse=True)
    
    return entities_cnt_sorted, tot_num_words

 

if __name__ == '__main__':


    # if len(sys.argv) != 3:
    #     print('Incorrect number of parameters entered. Please refer to the README on the input parameters.')
    #     exit()
    # else:
    #     doc_1 = sys.argv[1]
    #     doc_2 = sys.argv[2]
        
    #     # Get filenames and file extensions of the input docs
    #     doc_1_filename, doc_1_ext = os.path.splitext(sys.argv[1])[0], os.path.splitext(sys.argv[1])[1]
    #     doc_2_filename, doc_2_ext = os.path.splitext(sys.argv[2])[0], os.path.splitext(sys.argv[2])[1]
        
    file_docs = []
    
    # Lemmatizer object to remove stems from words
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Load first file to be compared against
    for base_file in all_files:
        doc_1_filename, doc_1_ext = os.path.splitext(base_file)[0], os.path.splitext(base_file)[1]
        doc_1 = doc_1_filename + doc_1_ext
        doc_1_cleaned = doc_1.replace(DATA_DIR + '\\', '')
                
        # Extract file extensions
        print(f'Reading first file ... {doc_1_cleaned}')
        # If .txt file:
        if doc_1_ext == '.txt':
            # with open(DATA_DIR + '\\' + doc_1) as f:
            with open(doc_1) as f:
                tokens = sent_tokenize(f.read())
                for line in tokens:
                    file_docs.append(line)

        # If .pdf file:
        elif doc_1_ext == '.pdf':
            file_docs, pdf_str_1 = tokenize_pdf_files(doc_1)
            pdf_entities_1, total_words_pdf_1  = named_entity_counts(pdf_str_1)
            # print(pdf_entities_1,total_words_pdf_1)
                
        else:
            raise TypeError
            print('A non-txt and pdf file detected. Please use only .txt or .pdf files')
            exit()
        
        gen_docs = []
        stop_words = set(stopwords.words('english')) 
        
        # Tokenize (and process) words for each sentence
        # Also lemmatize each word within the documents 
        gen_docs = [[wordnet_lemmatizer.lemmatize(w) for w in gensim.utils.simple_preprocess(text, min_len=3) 
                    if w not in stop_words ] for text in file_docs]
            
        # Remove 'blank' entries from the doc list
        for docs in gen_docs:
            # print(docs, int(len(docs)))
            if int(len(docs)) < 2:
                gen_docs.remove(docs)

        # print("Number of 'documents' in first file:", len(gen_docs))
        
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


        # Loop through second 'query' files and compare against the 'index' file (the 'first' file)
        for query_file in query_files:
            # Create Query Document - how similar is this query document to each document in the index
            file2_docs = []
            
            # Lemmatizer object to remove stems from words
            wordnet_lemmatizer_1 = WordNetLemmatizer()
            
            # Extract file extensions
            doc_2_filename, doc_2_ext = os.path.splitext(query_file)[0], os.path.splitext(query_file)[1]
            doc_2 = doc_2_filename + doc_2_ext
            doc_2_cleaned =  doc_2.replace(QUERY_DIR + '\\', '')
            
            print(f'Reading second file ... {doc_2_cleaned}')
            
            # If .txt file:
            if doc_2_ext == '.txt':
                # with open(DATA_DIR + '\\' + doc_2) as f:
                with open(doc_2) as f:
                    tokens = sent_tokenize(f.read())
                    for line in tokens:
                        file2_docs.append(line)

            # If .pdf file:
            elif doc_2_ext == '.pdf':
                file2_docs, pdf_str_2 = tokenize_pdf_files(doc_2)
                pdf_entities_2, total_words_pdf_2  = named_entity_counts(pdf_str_2)
                # print(pdf_entities_2, total_words_pdf_2)
            else:
                raise TypeError('A non-txt and pdf file detected. Please use only .txt or .pdf files')
                exit()

            print("Number of 'documents' in second file:", len(file2_docs))
            
            # print(file2_docs)

            # Array of averages (len = number of docs in the query)
            # Each entry in the list is the average similarity of the docs in the query doc compared to the corpus
            avg_sims = [] 

            for line in file2_docs:
                # tokenize & process words
                query_doc = gensim.utils.simple_preprocess(line, min_len=3)
                query_doc = [wordnet_lemmatizer_1.lemmatize(words) for words in query_doc if words not in stop_words]
                
                # print(query_doc, int(len(query_doc)))
                
                # If a blank row, skip to the next row
                if int(len(query_doc)) < 1:
                    continue
                
                # create bag of words
                query_doc_bow = dictionary.doc2bow(query_doc)
                # find similarity for each document
                query_doc_tf_idf = tf_idf[query_doc_bow]
                # print (document_number, document_similarity)
                # print('Comparing Result:', sims[query_doc_tf_idf]) 
                # calculate sum of similarities for each query doc
                sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
                # calculate average of similarity for each query doc
                avg = sum_of_sims / len(file_docs)
                # print average of similarity for each query doc
                print(f'avg: {sum_of_sims / len(file_docs)}')
                # add average values into array
                avg_sims.append(avg)  

                # print('AVERAGE SIMS: ' , avg_sims)    

            # calculate total average
            total_avg = ((np.sum(avg_sims, dtype=np.float)) / len(file2_docs))
            # results_total_avg.append(total_avg)

            # round the value and multiply by 100 to format it as percentage
            percentage_of_similarity = round(float(total_avg) * 100)
            # results_percentage_of_similarity.append(percentage_of_similarity)
            print('Similarity percentage: ' , percentage_of_similarity)
                
            results_df = results_df.append({'file_1': doc_1_cleaned, 
                'file_2': doc_2_cleaned, 
                'total_avg': total_avg,   
                'similarity_rating': percentage_of_similarity}, ignore_index=True)

    # if percentage is greater than 100
    if percentage_of_similarity >= 100:
        percentage_of_similarity = 100
    
    # Sort results DF by the total_avg (descending)
    results_df.sort_values('total_avg',inplace=True, ascending=False)

    # Customize pandas output to ensure no text is cut off
    pd.set_option("display.max_rows", None, "display.max_columns", 5, 'display.expand_frame_repr', False, 'display.max_colwidth', None)
    
    # Export dataframe to CSV file
    print(f'Saving results to {OUTPUT_DIR} directory ...')
    results_df.to_csv(OUTPUT_DIR + '//' + f'{today_str}.csv')

    

    

