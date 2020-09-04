import glob
import os
from datetime import datetime
import numpy as np
import pandas as pd
import spacy
import sys
from platform import system
from gensim_analysis.gensim_funcs import run_gensim_bow
from text_preprocessing.named_entity_extract import get_entity_log_freqs, get_entity_similarities
from text_preprocessing.preprocessing_funcs import get_clean_filename, read_file

# Load spacy model
nlp = spacy.load('en_core_web_lg')

# Get the current date for the filename
today = datetime.today()
today_str = today.strftime("%m-%d-%Y-%H%M%S")

# Directory variables (root dir, data dir, etc.)
ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(os.path.abspath(os.curdir), 'data')
QUERY_DIR = os.path.join(os.path.abspath(os.curdir), 'query')
OUTPUT_DIR = os.path.join(os.path.abspath(os.curdir), 'out')

# List of supported file extensions
if system() == 'Windows':
    supported_files = ["\*.pdf*", "\*.txt*"]
else:
    supported_files = ["/*.pdf*", "/*.txt*"]

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

# Check both folders to make sure valid files are in the directories
if len(all_files) == 0 or len(query_files) == 0:
    print('No documents located in ''data'' and/or ''query'' folder. Please add files.')
    exit()

# Initialize df to store comparison results
results_df = pd.DataFrame(columns=['file_1', 'file_2'])

if __name__ == '__main__':

    all_ents_with_no_vector = list()

    # Store "num_top_words" variable from input parameter
    try:
        num_top_words = int(sys.argv[1])
    except ValueError as wrong_param_type:
        num_top_words = 15
    except IndexError as index_of_of_range:
        num_top_words = 15

    # Flag for executing gensim BoW comparison
    try:
        gensim_flag = int(sys.argv[2])
    except IndexError as index_of_of_range:
        gensim_flag = 0
    except ValueError as wrong_param_type:
        gensim_flag = 0

    ############## Named Entity analysis ##############

    # Loop through both sets of files
    for base_file in all_files:
        doc_1_filename, doc_1_ext, doc_1, doc_1_cleaned = get_clean_filename(base_file)
        pdf_entities_1 = read_file(nlp, base_file)
        for query_file in query_files:
            doc_2_filename, doc_2_ext, doc_2, doc_2_cleaned = get_clean_filename(query_file)
            pdf_entities_2 = read_file(nlp, query_file)

            # Get similarity ratings between the entities in the two docs
            sim_ratings, num_duplicate_entities, ents_with_no_vector = get_entity_similarities(nlp, num_top_words,
                                                                                               pdf_entities_1,
                                                                                               pdf_entities_2)

            # Get frequency metrics for entities
            log_freq_prod = get_entity_log_freqs(nlp, num_top_words, pdf_entities_1, pdf_entities_2)
            # print(f'log_freq_prod: {log_freq_prod} , {len(log_freq_prod)}')

            # If all entities are duplicated between the two docs ...
            if num_duplicate_entities + len(ents_with_no_vector) >= num_top_words:
                print('ALL ENTITIES ARE DUPLICATES')
                similarity_score = 100
            else:
                similarity_score = (np.mean(sim_ratings) * 50) + ((num_duplicate_entities / int(num_top_words)) * 50)

            print('Similarity score: ', similarity_score)

            # Add similarity score to results list
            similiarity_rating_avg_cumul.append(similarity_score)

            for ent in ents_with_no_vector:
                all_ents_with_no_vector.append(ent.text)

            results_df = results_df.append({'file_1': doc_1_cleaned, 'file_2': doc_2_cleaned}
                                           , ignore_index=True)

    # Finalize output file
    print('List of all similarity ratings ... ', similiarity_rating_avg_cumul)

    # Add 'entity similarity rating' to the results dataframe
    results_df['entity_sim_rating'] = similiarity_rating_avg_cumul

    # Sort results DF by the similarity rating (descending)
    results_df.sort_values('entity_sim_rating', inplace=True, ascending=False)

    # Customize pandas output to ensure no text is cut off
    pd.set_option("display.max_rows", None, "display.max_columns", 5, 'display.expand_frame_repr', False,
                  'display.max_colwidth', None)

    # Create set of entities without word vectors
    ent_set = set(all_ents_with_no_vector)
    print(f'Ents with no vector ... {ent_set}')

    # Append list of entities without word vectors to a saved text file
    with open(OUTPUT_DIR + '//' + 'no_vector_entitites' + '//' 'no_vector_entities.txt', 'a') as filehandle:
        filehandle.writelines("%s\n" % ent for ent in ent_set)

    ############## Named Entity analysis (end) ##############

    ############## Gensim BoW analysis ##############

    # Run Gensim BoW script if flag is enabled
    if gensim_flag == 1:
        similarity_scores = run_gensim_bow()
        print(similarity_scores, len(similarity_scores))

        # Create column in df with similarity scores
        results_df['BOW_Comparison_score'] = similarity_scores

    else:
        pass

    ############## End Gensim BoW analysis (end) ##############

    # Export dataframe to CSV file
    print(f'Saving results to {OUTPUT_DIR} directory ...')
    results_df.to_csv(OUTPUT_DIR + '//' + f'{today_str}.csv')
