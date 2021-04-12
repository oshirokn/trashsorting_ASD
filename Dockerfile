From python:3.8.2-slim

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN pip3 install -r requirements.txt