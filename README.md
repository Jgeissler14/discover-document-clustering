# Document Clustering Code

This Python script is intended to analyze and classify documents in a large corpus, through cluster analysis, into groups of other similar documents.

A 'similarity threshold' - a value between 0 and 1 - can be set so that if a document is as similar to another document (i.e., has a similiarity rating greater than or equal to the similarity threshold), the documents are grouped together.

## Preparing the environment

Install the required packages in from the'requirements.txt' file via pip: "pip install  -r requirements.txt"
Please note that the script utilizes the Spacy package so please have both 'en_core_web_sm' and 'en_core_web_lg' available (e.g., "python -m spacy download en_core_web_lg").

### Assumptions for Input Files

1. The files in the directory are either .txt or .pdf files
2. The files are located in the "data" folder within the root directory

### Input parameters

1. Filename 1 (with file extension)
2. Filename 2 (with file extension)
3. Similarity threshold (TBD)

### Execute Command

CD to the working directory and run the following command to exeucte the script: python app.py [filename1] [filename2].
NOTE: The full filepath is not required as part of the input parameters. For example, if your file is called "filename.pdf", please enter just "filename.pdf".

### Output

Right now, the script prints output text (average similarities; cumulative similarity percentage) to the command line, as well as to a .csv file that is saved to the "out" directory within the root directory.
