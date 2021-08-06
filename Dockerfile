FROM python:3.7
WORKDIR /app
COPY pip3 install -r requirements.txt
EXPOSE 8080
COPY . /app
CMD streamlit run --server.port 8080 --server.enableCORS false app.py
