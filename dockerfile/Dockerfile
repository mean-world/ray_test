#FROM ubuntu:22.04 
FROM nvcr.io/nvidia/pytorch:22.04-py3
RUN apt update && apt install -y curl python3 pip

#kubectl 
RUN curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl"
RUN chmod +x ./kubectl && mv ./kubectl /usr/local/bin/kubectl

#python client
RUN pip3 install kubernetes
RUN pip3 install "ray[all]"
RUN pip3 install pytz
