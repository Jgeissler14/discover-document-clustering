# Document Clustering Code

This Python script is intended to analyze one or more documents from a large corpus and assign similarity ratings (from 0 to 100) for each pair of documents. A similarity score closer to 100 indicates the documents are very similar and scores closer to 0 indicate documents are dissimilar.

## Preparing the environment

    $ conda create -n <virtual-env-name>
    $ conda activate <virtual-env-name>

Install the required packages in from the 'requirements.txt' file via pip:
    $ pip install  -r requirements.txt

Please note that the script utilizes the Spacy package so please have the 'en_core_web_lg' spaCy model available
    $ python3 -m spacy download en_core_web_lg

The algorithm uses word vectors so the 'large' model is required.

### Assumptions for Input Files

1. The files are located in one of two directories in the root project directory: "data" and "query".
2. The files in the directories are either .txt or .pdf files

### Input parameters

1. The number of entities that you would like to be compared across documents. For example, if you would like to analyze the top 10 most commonly occurring entities throughout documents, please use "10" as the input parameter (e.g., "python app.py 10"). By default, the top 15 most frequently occurring entities will be compared across document pairs
2. "Bag of words" flag. Please enter a "1" as the second input parameter if you want to run a "bag of words" analysis in addition to a named-entity comparison with the document pairs. By default, this flag is turned off so only the "named entity" analysis will be conducted.
NOTE: All input parameters are optional and will use default values for the "top-K" named entities (15) and whether to include the "BoW" analysis with the script execution (0-off).

### Execute Command

CD to the working directory and run the following command to execute the script:

    python3 app.py <num_top_words> <bow_flag>

The script will execute and will extract the text from the files located in the "data" and "query" directories.

### Output

The script outputs a CSV file that contains a breakdown of each document pair that was compared, the named entity similarity score, and the "bag of words" similarity score. The script saves the file to the 'out' directory in the project directory

### Entities without word vectors

If a named entity is extracted from a doc but does not have an associated word vector in the spaCy model, these entities will be removed from the analysis and documented in a .txt file located in the 'out/no_vector_entities" folder
