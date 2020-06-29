import gensim
from nltk.stem import WordNetLemmatizer
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
import numpy as np
import os, glob
from nltk.corpus import stopwords
from text_preprocessing.preprocessing_funcs import get_clean_filename
from text_preprocessing.preprocessing_funcs import tokenize_pdf_files


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
similiarity_rating_avg_cumul = list()

# Find all valid files in the 'data' directory and append to 'all_files' list
for extension in supported_files:   
    for file in glob.glob(DATA_DIR + extension):
        all_files.append(file)

# Find all valid files in the 'query' directory and append to 'all_files_query' list
for extension in supported_files:   
    for file in glob.glob(QUERY_DIR + extension):
        query_files.append(file)


def run_gensim_bow():

	gensim_all_files = list()
	gensim_query_files = list()

	# Find all valid files in the 'data' directory and append to 'all_files' list
	for extension in supported_files:   
	    for file in glob.glob(DATA_DIR + extension):
	        gensim_all_files.append(file)

	# Find all valid files in the 'query' directory and append to 'all_files_query' list
	for extension in supported_files:   
	    for file in glob.glob(QUERY_DIR + extension):
	        gensim_query_files.append(file)
    
    # Lemmatizer object to remove stems from words
	wordnet_lemmatizer = WordNetLemmatizer()
 
	stop_words = set(stopwords.words('english')) 
	doc_1_gensim = list()
	doc_2_gensim = list()
	gensim_similarities = list()
    
            
	for doc_1 in all_files:
		doc_1_filename, doc_1_ext, doc_1, doc_1_cleaned = get_clean_filename(doc_1)
		doc_1_gensim.append(doc_1_cleaned)
		gensim_file_docs, pdf_str_1 = tokenize_pdf_files(doc_1)
  
		# Tokenize (and process) words for each sentence
		# Also lemmatize each word within the documents 
		gen_docs = [[wordnet_lemmatizer.lemmatize(w) for w in gensim.utils.simple_preprocess(text, min_len=3) 
		             if w not in stop_words ] for text in gensim_file_docs]

		### First file

		# Remove 'blank' entries from the doc list
		for docs in gen_docs:
		    # print(docs, int(len(docs)))
		    if int(len(docs)) < 2:
		        gen_docs.remove(docs)

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
		gensim_file2_docs = []

		for doc_2 in gensim_query_files:
			doc_2_filename, doc_2_ext, doc_2, doc_2_cleaned = get_clean_filename(doc_2)
			doc_2_gensim.append(doc_2_cleaned)
			gensim_file2_docs, pdf_str_2 = tokenize_pdf_files(doc_2)


			### Second file

			# Array of averages (len = number of docs in the query)
			# Each entry in the list is the average similarity of the docs in the query doc compared to the corpus
			avg_sims = [] 

			for line in gensim_file2_docs:
			    # tokenize & process words
			    query_doc = gensim.utils.simple_preprocess(line, min_len=3)
			    query_doc = [wordnet_lemmatizer.lemmatize(words) for words in query_doc if words not in stop_words]
			    
			    # If a blank row, skip to the next row
			    # print(query_doc, int(len(query_doc)))
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
			    avg = sum_of_sims / len(gensim_file_docs)
			    # print average of similarity for each query doc
			    # print(f'avg: {sum_of_sims / len(file_docs)}')
			    # add average values into array
			    avg_sims.append(avg) 
             

			# print('AVERAGE SIMS: ' , avg_sims)    

			# calculate total average
			total_avg = ((np.sum(avg_sims, dtype=np.float)) / len(gensim_file2_docs))

			# round the value and multiply by 100 to format it as percentage
			percentage_of_similarity = round(float(total_avg) * 100)
			# print('Similarity percentage: ' , percentage_of_similarity)

			# if percentage is greater than 100
			if percentage_of_similarity >= 100:
			    percentage_of_similarity = 100
			
			gensim_similarities.append(percentage_of_similarity)


	return gensim_similarities