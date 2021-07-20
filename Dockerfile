#FROM public.ecr.aws/lambda/python:3.8
FROM python:3.8
LABEL Dovel Technologies LLC "discover@doveltech.com"

WORKDIR /var/task

ADD ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install boto3
RUN python3 -m spacy download en_core_web_lg

ADD ./main.py .
ADD ./helper_functions.py .
ADD ./text_preprocessing ./text_preprocessing
ADD ./cluster.py .
ADD ./gensim_analysis ./gensim_analysis

CMD ["python", "./main.py"]
#CMD [ "main.handler" ]