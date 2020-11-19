# Document Clustering Code

This Python script is intended to analyze one or more documents from a large corpus and assign similarity ratings (from 0 to 100) for each pair of documents. A similarity score closer to 100 indicates the documents are very similar and scores closer to 0 indicate documents are dissimilar.

## Preparing the environment

    $ conda create -n clustering python=3.8
    $ conda activate clustering

Install the required packages in from the 'requirements.txt' file via pip:
    
    $ pip install  -r requirements.txt

Please note that the script utilizes the Spacy package so please have the 'en_core_web_lg' spaCy model available
    
    $ python3 -m spacy download en_core_web_lg

The algorithm uses word vectors so the 'large' model is required.

### Assumptions for Input Files

1. The files are located in the 'input' directory in the root project directory.
2. The files in the directory are either .txt or .pdf files.
3. The filenames of the files do not contain any spaces.

### Input parameters

1. The filename (with file extension) of the input document.
2. The filename (with file extension) of the "query" document.
3. "Bag of words" flag. Please enter a "1" as the second input parameter if you want to run a "bag of words" analysis in addition to a named-entity comparison with the document pairs.
4. The number of entities that you would like to be compared across documents. For example, if you would like to analyze the top 10 most commonly occurring entities throughout documents, please use "10" as the input parameter.

### Execute Command

CD to the working directory and run the following command to execute the script:

python app.py <filename_1> <filename_2> <num_top_words> <bow_flag>

The script will execute and will extract the text from the files located in the "data" and "query" directories.

### Output

The script outputs a CSV file that contains a breakdown of each document pair that was compared, the named entity similarity score, and the "bag of words" similarity score. The script saves the file to the 'out' directory in the project directory

### Entities without word vectors

If a named entity is extracted from a doc but does not have an associated word vector in the spaCy model, these entities will be removed from the analysis and documented in a .txt file located in the 'out/no_vector_entities" folder

### Auto-docker 

Auto-docker builds a container image and pushes the result to AWS ECR.

From the command line, checkout the latest code for this project and use auto-docker project to automate
the docker image build and deployment:

Skip the following steps if already in the project directory
```
    cd ~/github      # Your local git project directory 
    git clone https://github.com/DovelLabs/discover-document-clustering
    
    cd discover-document-clustering
    # If building a container from a branch other than master
    # git checkout autodocker   
```

Switch ot auto-docker project to build and deploy the container

```
    cd ~/github
    git clone https://github.com/DovelLabs/auto-docker
    cd auto-docker

    # The false means do not push to AWS, leave it blank to push to AWS ECR
    python main.py discover-document-clustering ../discover-document-clustering {version number|latest} {true|false} 
    
    # To run on your local docker, issue the following docker command
    # docker run 461136979341.dkr.ecr.us-east-1.amazonaws.com/discover-document-clustering
```
