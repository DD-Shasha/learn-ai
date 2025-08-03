# Dockerfile
FROM python:3.10

WORKDIR /app

COPY app/ /app/

RUN pip install -U langchain-community
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
