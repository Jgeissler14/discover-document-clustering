import logging
import os
import re
import zipfile
import tarfile
import boto3
from botocore.exceptions import ClientError
from shutil import copyfile


# Function to upload file to s3
def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

#create a copy of file from s3 to DOWNLOAD_DIR
def copy_from_s3(fi, DOWNLOAD_DIR):
    S3 = boto3.client('s3')
    # Use Regex to find the bucket and key matches
    regex = re.search('https://(.+?).s3.amazonaws.com/(.*)', fi)

    # s3 bucket name
    bucket_name = regex.group(1)
    # Path to file inside bucket
    key_name = regex.group(2)

    path, filename = os.path.split(key_name)
    input_file = os.path.join(DOWNLOAD_DIR, filename)
    S3.download_file(bucket_name, key_name, input_file)

    return input_file

#Check filetype and extract files if needed
def extract_files(filename, INPUT_DIR):
    S3 = boto3.client('s3')

    if filename.endswith(".zip"):
        #print("Zip file")
        with zipfile.ZipFile(filename, "r") as zip_file:
            zip_file.extractall(INPUT_DIR)
    elif filename.endswith(".tar"):
        #print("Tar file")
        with tarfile.open(filename, "r:") as tar_file:
            tar_file.extractall(INPUT_DIR)
    elif filename.endswith(".tar.gz"):
        #print("Tar gz file")
        with tarfile.open(filename, "r:gz") as tar_file:
            tar_file.extractall(INPUT_DIR)
    else:
       #print("Regular file")
        copyfile(filename, os.path.join(INPUT_DIR,os.path.basename(filename)))