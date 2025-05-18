FROM python:3.12.4

WORKDIR /app

COPY ./app.py /app/app.py
COPY ./Meta-Llama-3.1-8B-Instruct-bf16.gguf /app/Meta-Llama-3.1-8B-Instruct-bf16.gguf
# COPY ./requirements.txt /app/requirements.txt

# RUN pip install -r requirements.txt --no-cache-dir
RUN pip install Flask --no-cache-dir
RUN pip install llama-cpp-python --no-cache-dir

EXPOSE 5010

CMD ["python", "app.py"]
