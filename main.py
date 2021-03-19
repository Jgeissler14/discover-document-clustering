import glob
import itertools
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
today_str = today.strftime("%Y-%m-%d-%H%M%S")

# Create DIR variables
ROOT_DIR = os.path.abspath(os.getcwd())
OUTPUT_DIR = os.path.abspath('output')
INPUT_DIR = os.path.abspath('input')

# List of supported filetypes
supported_files = ["*.pdf", "*.txt"]


def main():
    # This script will be executed in a container in a batch environment.

    # ######### Parameters ##########
    # Do not pass variables on the command line, read all the required parameters
    # from the ENV variables. Discover UI will collect the parameters needed and set them as ENV variables at run time.
    import sys

    # Gensim flag parameter
    try:
        gensim_flag = int(sys.argv[1])
    except TypeError as gensim_flag_invalid:
        print('Invalid value provided for gensim_flag parameter. Exiting.')
        exit()
    except IndexError as gensim_flag_missing:
        gensim_flag = int(os.getenv("AD_GENSIM_FLAG", 0))

    # Num_top_words parameter
    try:
        num_top_words = int(sys.argv[2])
    except TypeError as top_words_invalid:
        print('Invalid value provided for num_top_words parameter. Exiting.')
        exit()
    except IndexError as num_top_words_missing:
        num_top_words = int(os.getenv("AD_TOPN_WORDS", 15))

    # Discover UI uses 'results.json' file to display the output to use
    output_results = {"data": [], "data_type": "generated"}

    # Initialize df to store comparison results
    results_df = pd.DataFrame(columns=['file_1', 'file_2'])

    # Initialize empty lists to store results
    all_ents_with_no_vector = list()
    similiarity_rating_avg_cumul = list()
    all_files_list = list()

    # Find all valid files in the 'input' directory and append to 'all_files_list' list
    for extension in supported_files:
        for idx, filepath in enumerate(glob.glob(os.path.join(INPUT_DIR, extension))):
            all_files_list.append(os.path.basename(filepath))

    # Create list of all document pairs (without replacement)
    res = [(file1, file2)
           for file1, file2 in itertools.combinations(all_files_list, 2)]

    # Loop through the list of filename pairs
    # TODO - Figure out a way to instantiate a spaCy object one for each of the files.
    # We're reading each file multiple times if there are multiple combinations with that file.

    for x in res:

        # Analyze the first file
        pdf_entities_1 = read_file(nlp, x[0])

        # Analyze the second file
        pdf_entities_2 = read_file(nlp, x[1])

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
            similarity_score = np.round((np.mean(sim_ratings) * 50) +
                                        ((num_duplicate_entities / int(num_top_words)) * 50), 2)

        print('Similarity score: ', similarity_score)

        # Add similarity score to results list
        similiarity_rating_avg_cumul.append(similarity_score)

        # Add entities without a vector to a list
        for ent in ents_with_no_vector:
            all_ents_with_no_vector.append(ent.text)

        # Create results_df file
        results_df = results_df.append(
            {'file_1': x[0], 'file_2': x[1]}, ignore_index=True)

    # Finalize output file
    print('List of all similarity ratings ... ', similiarity_rating_avg_cumul)

    # Add 'entity similarity rating' to the results dataframe
    results_df['entity_sim_rating'] = similiarity_rating_avg_cumul

    # Sort results DF by the similarity rating (descending)
    results_df.sort_values('entity_sim_rating', inplace=True, ascending=False)

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

        # Generate gensim BOW scores for each pair
        gensim_sim_scores_list = [run_gensim_bow(x[0], x[1]) for x in res]

        # Flatten the list of gensim scores
        gensim_sim_scores_list_flat = list(
            itertools.chain(*gensim_sim_scores_list))

        # Create column in df with similarity scores
        results_df['bow_comparison_score'] = gensim_sim_scores_list_flat

    # ############# End Gensim BoW analysis (end) ##############

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
        print(str(tb))
        with open(os.path.join(OUTPUT_DIR, "results.json"), "w+") as f:
            print("Writing errors in results.json file")
            json.dump({
                "data_type": "generated",
                "data": [
                    {"error": str(tb), "title": "Error"}
                ]
            }, f)
            f.close()
