# Document Clustering Code

This Python script is intended to analyze and classify documents in a large corpus, through cluster analysis, into groups of other similar documents.

A 'similarity threshold' - a value between 0 and 1 - can be set so that if a document is as similar to another document (i.e., has a similiarity rating greater than or equal to the similarity threshold), the documents are grouped together.

## Preparing the environment

Install the required packages in from the'requirements.txt' file via pip: "pip install  -r requirements.txt"

### Assumptions for Input Dataset

1. The files in the directory are .txt files
2. The files are located in the "data" folder within the root directory

### Input parameters

1. Directory of files

### Execute Command

CD to the working directory and run the following command to exeucte the script: python app.py [directory 1] [directory 2] 


### Output


