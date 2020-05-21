import numpy as np
import nltk, gensim, glob, os
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
from datetime import date

# Get the current date for the filename
today = date.today()
today = today.strftime("%m-%d-%Y")

# Directory variables (root dir, data dir, etc.)
ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(os.path.abspath(os.curdir), 'data')


# Get list of documents in location #1

# Get list of documents in location #2

# Function to get list of documents in a provided directory
def get_docs(location= os.curdir):
    dir_files = [file for file in glob.glob(DATA_DIR + "\*.txt*")]
    return dir_files



if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 3:
        print('Incorrect number of parameters entered. Please refer to the README on the input parameters.')
        exit()
    else:
        doc_1 = sys.argv[1]
        doc_2 = sys.argv[2]
        
file_docs = []

# Open the document file
with open(DATA_DIR + '\\' + doc_1) as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)

print("Number of documents:", len(file_docs))

# Tokenize words for each sentence
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in file_docs]

# Create gensim dictionary with ID as the key, and word token as the value
dictionary = gensim.corpora.Dictionary(gen_docs)
# print(dictionary.token2id)

### Create a bag of words corpus, passing the tokenized list of words to the Dictionary.doc2bow()
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

# TFIDF - down weights tokens (words) that appears frequently across documents
# Words that occur more frequently across the documents get smaller weights
tf_idf = gensim.models.TfidfModel(corpus)
for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

# Create similarity measure object
index_tmpfile = get_tmpfile("wordindex")
sims = gensim.similarities.Similarity(index_tmpfile,tf_idf[corpus],
                                        num_features=len(dictionary))

# Create Query Document - how similar is this query document to each document in the index
file2_docs = []

# Open the document file
with open(DATA_DIR + '\\' + doc_2) as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file2_docs.append(line)

#update an existing dictionary and create bag of words
for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc) 

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
tf_idf = gensim.models.TfidfModel(corpus)

# Document similarities - query document against indexed documents

# Perform a similarity query against the corpus
query_doc_tf_idf = tf_idf[query_doc_bow]

# Cosine measure returns similarities in the range (greater value --> the more similar).
print('Comparing Result:', sims[query_doc_tf_idf]) 


### Calculate average similarity of query document
sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
print(sum_of_sims)

# Percentage similarity
percentage_of_similarity = round(float((sum_of_sims / len(file_docs)) * 100))
print(f'Average similarity float: {float(sum_of_sims / len(file_docs))}')
print(f'Average similarity percentage: {float(sum_of_sims / len(file_docs)) * 100}')
print(f'Average similarity rounded percentage: {percentage_of_similarity}')

avg_sims = [] # array of averages

# for line in query documents
for line in file2_docs:
    # tokenize words
    query_doc = [w.lower() for w in word_tokenize(line)]
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
        
# calculate total average
total_avg = np.sum(avg_sims, dtype=np.float)
total_avg = ((np.sum(avg_sims, dtype=np.float)) / len(file2_docs))

# round the value and multiply by 100 to format it as percentage
percentage_of_similarity = round(float(total_avg) * 100)

# if percentage is greater than 100
if percentage_of_similarity >= 100:
	percentage_of_similarity = 100


