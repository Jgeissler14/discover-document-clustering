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

no_vector_ents = list()


# Find all valid files in the 'data' directory and append to 'all_files' list
for extension in supported_files:   
    for file in glob.glob(DATA_DIR + extension):
        all_files.append(file)

# Find all valid files in the 'query' directory and append to 'all_files_query' list
for extension in supported_files:   
    for file in glob.glob(QUERY_DIR + extension):
        query_files.append(file)

# Check both folders to make sure valid files are in the directories
if len(all_files) == 0 or len(query_files) == 0:
    print('No documents located in ''data'' and/or ''query'' folder. Please add files.')
    exit()

# Initialize df to store comparison results   
results_df = pd.DataFrame(columns=['file_1','file_2','total_avg','similarity_rating'])

# Read in PDF file and return list of unprocessed docs (sentences) from the file
def tokenize_pdf_files(pdf_filename):
    raw_text = textract.process(pdf_filename, encoding='utf-8')
    str_raw_text = raw_text.decode('utf-8')
    pdf_token = sent_tokenize(str_raw_text)
    
    return pdf_token, str_raw_text

# Create spacy object
nlp = spacy.load('en_core_web_lg')
# nlp = spacy.load('en_core_web_sm')

similiarity_rating_avg_cumul = list()

def get_named_entity_counts(raw_text):
    # Create spacy doc
    doc = nlp(raw_text)
    
    # Store the number of words in the doc
    tot_num_words = len(doc)

    # Create list of named entities along with their labels
    doc_entities = [[re.sub('  ', ' ', re.sub('\r',' ',re.sub('\n', ' ', ent.text))).strip().lower(), ent.label_, ] 
                    for ent in doc.ents if ent.text.isnumeric() == False and ent.text.startswith('https://') == False] 
    
    # Create defaultdict to list the entities and their counts
    entities_cnt = defaultdict(int)
    
    # Create frequency count for each of the entities
    for entity, ent_label in doc_entities:
        entities_cnt[entity] +=1

    # Sort the entities list by the number of occurrences 
    entities_cnt_sorted = sorted(entities_cnt.items(), key=lambda x: x[1], reverse=True)
    # print(entities_cnt_sorted)
    
    # Get the entity frequency list (including counts, entity cnt / total doc length, & log frequency 
    ent_doc_freq_list = [[entity[0],  entity[1], entity[1]/tot_num_words, np.log(1 + int(entity[1]))] 
                         for entity in entities_cnt_sorted]
 
    # print(ent_doc_freq_list)
        
    # Also get a simple list of just the entity names
    extracted_entities = [entity[0] for entity in entities_cnt_sorted]
    entity_cnt =    [entity[1] for entity in entities_cnt_sorted]
    entity_freq = [entity[1]/tot_num_words for entity in entities_cnt_sorted]
    entity_log_freq = [np.log(1 + int(entity[1])) for entity in entities_cnt_sorted]
    
    # print(extracted_entities) 

    return ent_doc_freq_list

# Function to create a list of similarity ratings for the most common entities between docs
def get_entity_similarities(pdf_entities_1, pdf_entities_2): 
    
    # Initialize empty list to house entity similarity comparisons
    entity_similarities = list() 
    num_duplicate_entities = int()
    
    doc_1_entities = [term_1 for term_1, _ , _ , _ in pdf_entities_1[:10]]
    doc_2_entities = [term_2 for term_2, _ , _ , _ in pdf_entities_2[:10]]

    # Create lists of doc objects in bulk with nlp.pipe
    ents_1 = list(nlp.pipe(doc_1_entities))
    ents_2 = list(nlp.pipe(doc_2_entities))
    # print(ents_1, ents_2)
    
    idx_list = list()
    idy_list = list()
    duplicated_ents = list()
    
    
    # Remove entities without word vectors:
    for idx, x in enumerate(ents_1):
    # Check if entity has a vector; if not, remove it from the list
        if x.vector_norm == 0:
            print('*** no vector found', x)
            ents_1.pop(idx)
            no_vector_ents.append(x)
    for idy, y in enumerate(ents_2):
        # Check if entity has a vector; if not, remove it from the list
        if y.vector_norm == 0:
            print('*** no vector found', y)
            ents_2.pop(idy)
            no_vector_ents.append(y)
    
    # Find duplicate entities from both docs
    for idx, x in enumerate(ents_1):
        for idy, y in enumerate(ents_2):
            # Check if entities are duplicates       
            if x.text == y.text and x.similarity(y) == 1.0:
                # If so, add the indexes to two lists
                idx_list.append(idx)
                idy_list.append(idy)
                duplicated_ents.append(x)

    # Reverse the sorting
    idx_list.sort(reverse=True)
    idy_list.sort(reverse=True)
    
    num_duplicate_entities = len(duplicated_ents)
    
    print('no_vector_ents ... ', no_vector_ents)
    print('duplicated_ents ... ', duplicated_ents, len(duplicated_ents))
    
    # Loop through both lists to get similarity ratings between the two lists of entities
    for f in ents_1:
        for b in ents_2:
            # print(f, b, f.similarity(b))
            entity_similarities.append(f.similarity(b))
    
    return entity_similarities, num_duplicate_entities, no_vector_ents

# Function to get the log frequency products between the two lists 
def get_entity_log_freqs(pdf_entities_1, pdf_entities_2):    
    
    # Initialize empty list to house entity log_frequency (normalized) values
    entity_log_freqs = list()
    
    doc_1_log_freqs = [log_freq_1 for _, _ , _ , log_freq_1 in pdf_entities_1[:10]]
    doc_2_log_freqs = [log_freq_2 for _, _ , _ , log_freq_2 in pdf_entities_2[:10]]
    
    for x in doc_1_log_freqs:
        for y in doc_2_log_freqs:
            entity_log_freqs.append(x * y)
    
    return entity_log_freqs
            

if __name__ == '__main__':
    
    # Store "num_top_words" variable from input parameter
    num_top_words = sys.argv[1]
    
    # Create list to store docs from the data file  
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
            pdf_entities_1  = get_named_entity_counts(pdf_str_1)
            # print(pdf_entities_1)
                
        else:
            raise TypeError
            print('A non-txt and pdf file detected. Please use only .txt or .pdf files')
            exit()
        
        
        # Compare doc similarity (without named entities)
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
                pdf_entities_2 = get_named_entity_counts(pdf_str_2)
                # print(pdf_entities_2)
            else:
                raise TypeError('A non-txt and pdf file detected. Please use only .txt or .pdf files')
                exit()
            
            # Array of averages (len = number of docs in the query)
            # Each entry in the list is the average similarity of the docs in the query doc compared to the corpus
            avg_sims = [] 

            for line in file2_docs:
                
                # tokenize & process words
                query_doc = gensim.utils.simple_preprocess(line, min_len=3)
                query_doc = [wordnet_lemmatizer_1.lemmatize(words) for words in query_doc if words not in stop_words]
                
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
                
                # print(f'avg: {sum_of_sims / len(file_docs)}')
                
                # add average values into array
                avg_sims.append(avg)  

                # print('AVERAGE SIMS: ' , avg_sims)    

            # calculate total average
            total_avg = ((np.sum(avg_sims, dtype=np.float)) / len(file2_docs))
            # results_total_avg.append(total_avg)

            # round the value and multiply by 100 to format it as percentage
            percentage_of_similarity = round(float(total_avg) * 100)
            # results_percentage_of_similarity.append(percentage_of_similarity)
            
            # print('Similarity percentage: ' , percentage_of_similarity)
            
            # Append results to a dataframe
            results_df = results_df.append({'file_1': doc_1_cleaned, 
                'file_2': doc_2_cleaned, 
                'total_avg': total_avg,   
                'similarity_rating': percentage_of_similarity}
                                           , ignore_index=True)
        
        
        ### Entity comparison ###   
        
        # Get similarity ratings between the entities in the two docs
        sim_ratings, num_duplicate_entities, ents_with_no_vector = get_entity_similarities(pdf_entities_1, pdf_entities_2)
        # print(f'sim_ratings ... {sim_ratings}, {len(sim_ratings)}')
        
        # Get frequency metrics for entities
        log_freq_prod = get_entity_log_freqs(pdf_entities_1, pdf_entities_2)
        # print(f'log_freq_prod: {log_freq_prod} , {len(log_freq_prod)}')

        # Calculate similarity score of the two docs
        if len(sim_ratings) == 0:
            print('ALL ENTITIES WERE DUPLICATED')
            similarity_score = ((np.mean(sim_ratings) * 50) + (50))
        else:
            similarity_score = (np.mean(sim_ratings) * 50) + ((num_duplicate_entities/int(num_top_words)) * 50)
            
        print('Similarity score: ', similarity_score)

        # Add similarity score to results list
        similiarity_rating_avg_cumul.append(similarity_score)


    # Finalize output file
    print('List of all similarity ratings ... ', similiarity_rating_avg_cumul)
    
    # Add 'entity similarity rating' to the results dataframe
    results_df['entity_sim_rating'] = similiarity_rating_avg_cumul
    
    # Sort results DF by the similarity rating (descending)
    results_df.sort_values('entity_sim_rating',inplace=True, ascending=False)

    # Customize pandas output to ensure no text is cut off
    pd.set_option("display.max_rows", None, "display.max_columns", 5, 'display.expand_frame_repr', False, 'display.max_colwidth', None)
    
    # Export dataframe to CSV file
    print(f'Saving results to {OUTPUT_DIR} directory ...')
    results_df.to_csv(OUTPUT_DIR + '//' + f'{today_str}.csv')
    
    # Get list of entity names that don't have spaCy vector
    
    ent_texts = [ent.text for ent in ents_with_no_vector]
    ent_set = set(ent_texts)
    print(f'Ents with no vector ... {ent_set}')
    

