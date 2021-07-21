FROM public.ecr.aws/lambda/python:3.8
#FROM python:3.8
LABEL Dovel Technologies LLC "discover@doveltech.com"

WORKDIR /var/task

ADD ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install boto3

ENV NLTK_DATA=/var/task/nltk_data

ADD ./main.py .
ADD ./helper_functions.py .
ADD ./text_preprocessing ./text_preprocessing
ADD ./cluster.py .
ADD ./gensim_analysis ./gensim_analysis
ADD ./tokenizers ./tokenizers

#CMD ["python", "./main.py"]
CMD [ "main.handler" ]