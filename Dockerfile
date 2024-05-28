FROM huggingface/accelerate-gpu:latest

WORKDIR /code
COPY . .

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install nano -y
RUN apt install htop
RUN apt-get install libaio-dev -y

RUN pip install --upgrade pip
RUN pip install datasets transformers accelerate
RUN pip3 install deepspeed
RUN pip install wandb