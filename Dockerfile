FROM ubuntu:20.04
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-dev

RUN pip install poetry
RUN pip install poethepoet

WORkDIR /text_affinity
COPY . /text_affinity

RUN poetry install
EXPOSE 8000
CMD ['poetry run uvicorn webapp.text_affinity_api:app']