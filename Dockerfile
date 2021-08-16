FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["app.py"]
