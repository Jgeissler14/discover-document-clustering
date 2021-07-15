import json
import os
import re
from re import search
import glob
import boto3
import pandas
from shutil import copyfile
from helper_functions import upload_file
from helper_functions import copy_from_s3
from helper_functions import extract_files
from cluster import cluster


# define location of supporting data files
DATA_DIR = os.path.join(os.getcwd(), 'data')

#Lambda only supports writing files to /tmp
#Determine if workdir is /var/task which implies running in a container
if os.getcwd() == '/var/task':
    OUTPUT_DIR = '/tmp/output'
    DOWNLOAD_DIR = '/tmp/download'
    INPUT_DIR = '/tmp/input'
else:
    OUTPUT_DIR = './tmp/output'
    DOWNLOAD_DIR = './tmp/download'
    INPUT_DIR = './tmp/input'

#check if output and download dirs exist and create them if not
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
if not os.path.isdir(DOWNLOAD_DIR):
    os.mkdir(DOWNLOAD_DIR)
if not os.path.isdir(INPUT_DIR):
    os.mkdir(INPUT_DIR)

AD_S3_UPLOAD_BUCKET = os.getenv("AD_S3_UPLOAD_BUCKET", '')
AD_S3_UPLOAD_FOLDER = os.getenv("AD_S3_UPLOAD_FOLDER", '')

S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET", '')
S3_DATA_FOLDER = os.getenv("S3_DATA_FOLDER", '')

results_json_fn = "results.json"

#separate files by a comma in a single string if using multiple files.
input_file_list = ""
#input_file_list = "sample_1.csv,https://discover-s3-testing.s3.amazonaws.com/sample_2.csv,https://discover-s3-testing.s3.amazonaws.com/random/sample_3.csv"

S3 = boto3.client('s3')

#Lambda handler function to call process function
def handler(event, lambda_context):
    try:
        process(event=None, lambda_context=None)
        return("Success")
    except Exception as e:
        with open(os.path.join(OUTPUT_DIR, results_json_fn), "w+") as f:
            print("Writing errors in results.json file")
            json.dump({
                "data_type": "generated",
                "data": [
                    {"error": str(e), "title": "Error"}
                ]
            }, f)
            f.close()
        print(str(e))

        # Upload to S3 if needed
        if AD_S3_UPLOAD_BUCKET != '':
            if AD_S3_UPLOAD_FOLDER != '':
                upload_file(os.path.join(OUTPUT_DIR, results_json_fn), AD_S3_UPLOAD_BUCKET, AD_S3_UPLOAD_FOLDER + '/' + results_json_fn)
            else:
                upload_file(os.path.join(OUTPUT_DIR, results_json_fn), AD_S3_UPLOAD_BUCKET, results_json_fn)
    
        print(results_json_fn)
        return(str(e))
    
def process(event, lambda_context):
    # This script will be executed in a container in a batch environment.

    # ######### Parameters ##########
    # Do not pass variables on the command line, read all the required parameters
    # from the ENV variables. Discover UI will collect the parameters needed and set them as ENV variables
    # at run time.

    # Example: Read a float value for threshold and default to 0.75 if missing
    # threshold = float(os.getenv("AD_THRESHOLD", 0.75))

    input_files = os.getenv("AD_INPUT_NAME", input_file_list).split(',')
    print(input_files)

    # Discover UI uses 'results.json' file to display the output to use
    # For information on results.json format see: ???
    results_data = {"data": [], "data_type": "generated"}

    # Results object
    results_dict = {}

    #Loop through all files and copy them to DOWNLOADD_DIR
    #Extract files if needed to INPUT_DIR, if not copy them from DOWNLOAD_DIR to INPUT_DIR 
    for fi in input_files:
        print(fi)
        # Create a copy of file in DOWNLOAD_DIR
        # Extract files if neccessary to INPUT_DIR
        if search("s3.amazonaws.com", fi):
            filename = copy_from_s3(fi, DOWNLOAD_DIR)
            extract_files(filename, INPUT_DIR)
        #If not on s3, assume its local
        else:
            copyfile(fi, os.path.join(DOWNLOAD_DIR, fi))
            extract_files(os.path.join(DOWNLOAD_DIR, fi), INPUT_DIR)

    cluster()

     # Upload results json to S3
    if AD_S3_UPLOAD_BUCKET != '':
        for output_file in glob.glob(os.path.join(OUTPUT_DIR, "*")):
            if AD_S3_UPLOAD_FOLDER != '':
                upload_file(output_file, AD_S3_UPLOAD_BUCKET,  AD_S3_UPLOAD_FOLDER + '/' + os.path.basename(output_file))
            else:
                upload_file(output_file, AD_S3_UPLOAD_BUCKET, os.path.basename(output_file))

if __name__ == '__main__':
    handler(event=None, lambda_context=None)
