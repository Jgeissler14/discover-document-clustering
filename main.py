import glob
import json
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import spacy

from gensim_analysis.gensim_funcs import run_gensim_bow
from text_preprocessing.named_entity_extract import (get_entity_log_freqs,
                                                     get_entity_similarities)
from text_preprocessing.preprocessing_funcs import read_file

# Load spacy model
nlp = spacy.load('en_core_web_lg')

# Get the current date for the filename
today = datetime.today()
today_str = today.strftime("%m-%d-%Y-%H%M%S")

ROOT_DIR = os.path.abspath(os.getcwd())
OUTPUT_DIR = os.path.abspath('output')
INPUT_DIR = os.path.abspath('input')
# QUERY_DIR = os.path.abspath('query')

supported_files = ["*.pdf", "*.txt"]
# List to gather all filenames in the 'data' directory
all_files = list()
query_files = list()
similiarity_rating_avg_cumul = list()



def main():
    # This script will be executed in a container in a batch environment.

    # Initialize df to store comparison results
    results_df = pd.DataFrame(columns=['file_1', 'file_2'])

    all_ents_with_no_vector = list()

    # ######### Parameters ##########
    # Do not pass variables on the command line, read all the required parameters
    # from the ENV variables. Discover UI will collect the parameters needed and set them as ENV variables
    # at run time.
    import sys
    
    try:
        input_file_param, query_file_param = str(
            sys.argv[1]), str(sys.argv[2])
    except IndexError as missing_param:
        input_file_param = str(os.getenv("AD_INPUT_FILE"))
        query_file_param = str(os.getenv("AD_INPUT_QUERY"))
        
    # Gensim flag parameter
    try:
        gensim_flag = int(sys.argv[3])
    except TypeError as gensim_flag_invalid:
        print('Invalid value provided for gensim_flag parameter. Exiting.')
        exit()
    except IndexError as gensim_flag_missing:
        gensim_flag = int(os.getenv("AD_GENSIM_FLAG", 0))

    # Num_top_words parameter
    try:
        num_top_words = int(sys.argv[4])
    except TypeError as top_words_invalid:
        print('Invalid value provided for num_top_words parameter. Exiting.')
        exit()
    except IndexError as num_top_words_missing:
        num_top_words = int(os.getenv("AD_TOPN_WORDS", 15))
    
    # num_top_words = int(os.getenv("AD_TOPN_WORDS", 15))
    # print(f'Num Top Words={num_top_words}')
    
    
    # Example: Read a float value for threshold and default to 0.75 if missing
    # threshold = float(os.getenv("AD_THRESHOLD", 0.75))

    # Discover UI uses 'results.json' file to display the output to use
    # For information on results.json format see: ???
    output_results = {"data": [], "data_type": "generated"}

    # Results object
    results_dict = {}

    # Loop through both sets of files
    # for base_file in all_files:

    print(f'input_file_param: {input_file_param}')
    
    pdf_entities_1 = read_file(nlp, input_file_param)
    print(pdf_entities_1)
    
    # for query_file in query_files:

    pdf_entities_2 = read_file(nlp, query_file_param)
    print(pdf_entities_2)
   
    # Get similarity ratings between the entities in the two docs

    sim_ratings, num_duplicate_entities, ents_with_no_vector = get_entity_similarities(nlp, num_top_words,
                                                                                        pdf_entities_1,
                                                                                        pdf_entities_2)

    # Get frequency metrics for entities

    log_freq_prod = get_entity_log_freqs(
        nlp, num_top_words, pdf_entities_1, pdf_entities_2)
    # print(f'log_freq_prod: {log_freq_prod} , {len(log_freq_prod)}')

    # If all entities are duplicated between the two docs ...
    if num_duplicate_entities + len(ents_with_no_vector) >= num_top_words:
        print('ALL ENTITIES ARE DUPLICATES')
        similarity_score = 100
    else:
        similarity_score = (np.mean(sim_ratings) * 50) + \
            ((num_duplicate_entities / int(num_top_words)) * 50)

    print('Similarity score: ', similarity_score)

    # Add similarity score to results list
    similiarity_rating_avg_cumul.append(similarity_score)

    for ent in ents_with_no_vector:
        all_ents_with_no_vector.append(ent.text)

    results_df = results_df.append(
        {'file_1': input_file_param, 'file_2': query_file_param}, ignore_index=True)

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
