# Document Clustering Code

This Python script is intended to analyze and classify documents in a large corpus, through cluster analysis, into groups of other similar documents.

(under construction) A 'similarity threshold' - a value between 0 and 1 - can be set so that if a document is as similar to another document (i.e., has a similiarity rating greater than or equal to the similarity threshold), the documents are grouped together.

## Preparing the environment

Install the required packages in from the'requirements.txt' file via pip: "pip install  -r requirements.txt"
Please note that the script utilizes the Spacy package so please have both 'en_core_web_sm' and 'en_core_web_lg' available (e.g., "python -m spacy download en_core_web_lg").

### Assumptions for Input Files

1. The files are located in one of two directories in the root project directory: "data" and "query".
2. The files in the directories are either .txt or .pdf files

### Input parameters

None

### Execute Command

CD to the working directory and run the following command to exeucte the script: "python app.py". The script will execute and will extract the text from the files located in the "data" and "query" directories.

### Output

Right now, the script prints output text (average similarities; cumulative similarity percentage) to the command line, as well as to a .csv file that is saved to the "out" directory within the root directory.
