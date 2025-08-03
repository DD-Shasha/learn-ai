FROM python:3.10

WORKDIR /app

COPY app/ /app/

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
