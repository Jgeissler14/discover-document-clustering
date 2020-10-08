import glob
import json
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import spacy

from gensim_analysis.gensim_funcs import run_gensim_bow
# Load spacy model
from text_preprocessing.named_entity_extract import get_entity_similarities, get_entity_log_freqs
from text_preprocessing.preprocessing_funcs import read_file

nlp = spacy.load('en_core_web_lg')

# Get the current date for the filename
today = datetime.today()
today_str = today.strftime("%m-%d-%Y-%H%M%S")

ROOT_DIR = os.path.abspath(os.getcwd())
OUTPUT_DIR = os.path.abspath('output')
INPUT_DIR = os.path.abspath('input/input')
QUERY_DIR = os.path.abspath('input/query')

supported_files = ["*.pdf", "*.txt"]
# List to gather all filenames in the 'data' directory
all_files = list()
query_files = list()
similiarity_rating_avg_cumul = list()

# Find all valid files in the 'data' directory and append to 'all_files' list
print("get input files")
for extension in supported_files:
    for file in glob.glob(os.path.join(INPUT_DIR, extension)):
        print(file)
        all_files.append(file)

# Find all valid files in the 'query' directory and append to 'all_files_query' list
print("get query files")
for extension in supported_files:
    for file in glob.glob(os.path.join(QUERY_DIR, extension)):
        print(file)
        query_files.append(file)

# Check both folders to make sure valid files are in the directories
if len(all_files) == 0 or len(query_files) == 0:
    print('No documents located in ''input'' and/or ''query'' folder. Please add files.')
    exit()


def main():
    # This script will be executed in a container in a batch environment.

    # Initialize df to store comparison results
    results_df = pd.DataFrame(columns=['file_1', 'file_2'])

    all_ents_with_no_vector = list()

    # ######### Parameters ##########
    # Do not pass variables on the command line, read all the required parameters
    # from the ENV variables. Discover UI will collect the parameters needed and set them as ENV variables
    # at run time.

    # Example: Read a float value for threshold and default to 0.75 if missing
    # threshold = float(os.getenv("AD_THRESHOLD", 0.75))

    num_top_words = int(os.getenv("AD_TOPN_WORDS", 15))
    print(f'Num Top Words={num_top_words}')

    gensim_flag = int(os.getenv("AD_GENSIM_FLAG", 0))

    # Discover UI uses 'results.json' file to display the output to use
    # For information on results.json format see: ???
    output_results = {"data": [], "data_type": "generated"}

    # Results object
    results_dict = {}

    # Loop through both sets of files
    for base_file in all_files:

        # doc_1_filename, doc_1_ext, doc_1, doc_1_cleaned = get_clean_filename(base_file)
        doc_1_cleaned = os.path.basename(base_file)

        pdf_entities_1 = read_file(nlp, base_file)
        for query_file in query_files:
            # doc_2_filename, doc_2_ext, doc_2, doc_2_cleaned = get_clean_filename(query_file)
            doc_2_cleaned = os.path.basename(query_file)
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
    no_vector_filename = 'no_vector_entities.txt'
    with open(os.path.join(OUTPUT_DIR, no_vector_filename), 'a') as filehandle:
        filehandle.writelines("%s\n" % ent for ent in ent_set)

    # ############# Named Entity analysis (end) ##############

    # ############# Gensim BoW analysis ##############
    # Run Gensim BoW script if flag is enabled
    if gensim_flag == 1:
        similarity_scores = run_gensim_bow()
        print(similarity_scores, len(similarity_scores))

        # Create column in df with similarity scores
        results_df['BOW_Comparison_score'] = similarity_scores
    else:
        pass

    # ############# End Gensim BoW analysis (end) ##############

    # Export dataframe to CSV file
    print(f'Saving results to {OUTPUT_DIR} directory ...')
    output_filename = f'result_{today_str}.csv'
    results_df.to_csv(os.path.join(OUTPUT_DIR, output_filename))

    output_results["data"].append({
        "title": 'Document Similarity Results',
        "filename": output_filename,
        "link": os.path.join(OUTPUT_DIR, output_filename),
    })

    output_results["data"].append({
        "filename": no_vector_filename,
        "text": os.path.join(OUTPUT_DIR, no_vector_filename),
    })

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w+") as f:
        print("Writing results.json file")
        json.dump(output_results, f)
        f.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        with open(os.path.join(OUTPUT_DIR, "results.json"), "w+") as f:
            print("Writing errors in results.json file")
            json.dump({
                "data_type": "generated",
                "data": [
                    {"error": str(tb), "title": "Error"}
                ]
            }, f)
            f.close()
