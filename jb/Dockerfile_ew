#FROM python:3.9-slim
FROM python:3.9

WORKDIR /app

COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update
#RUN apt install -y wget
#RUN apt install -y --fix-missing g++

COPY service/app.py .
COPY service/fits.npy .
RUN mkdir static
#RUN mv spinner.css static/

COPY src/settings.py .
COPY src/demographics_settings.py .
COPY src/measles.py .
COPY src/sir_numpy_c.py .
COPY src/sir_numpy.py .
COPY src/report.py .
COPY src/update_ages.cpp .
RUN g++ -shared -fPIC -O3 -march=native -flto -fpermissive -fopenmp update_ages.cpp -o update_ages.so
COPY src/model_numpy/eula.py model_numpy/ 

RUN wget https://packages.idmod.org:443/artifactory/idm-data/laser/engwal_modeled.csv.gz
RUN wget https://packages.idmod.org:443/artifactory/idm-data/laser/attraction_probabilities.csv.gz
RUN wget https://packages.idmod.org:443/artifactory/idm-data/laser/cities.csv
RUN wget https://packages.idmod.org:443/artifactory/idm-data/laser/cbrs_ew.csv
RUN gunzip attraction_probabilities.csv.gz

COPY src/post_proc.py .
COPY data/pops.csv .

CMD ["python", "app.py"]

