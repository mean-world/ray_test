FROM ubuntu:22.04 
#cpu only
#FROM nvcr.io/nvidia/pytorch:22.04-py3 #gpu version
RUN apt update && apt install -y curl python3 pip
RUN apt update && apt install -y git

#kubectl 
RUN curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl"
RUN chmod +x ./kubectl && mv ./kubectl /usr/local/bin/kubectl

#python client
RUN pip3 install kubernetes
#ray version 2.9.3
RUN pip3 install "ray[all]"
#RUN python -m pip install -U pydantic spacy
RUN pip3 install torch pytz mlflow tqdm scikit-learn  gitpython 
RUN pip3 install yfinance


