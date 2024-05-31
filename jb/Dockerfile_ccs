#FROM python:3.9-slim
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update
#RUN apt install -y wget
#RUN apt install -y --fix-missing g++

COPY app.py .
RUN mkdir static
#RUN mv spinner.css static/

COPY settings.py .
COPY demographics_settings.py .
COPY tlc.py .
COPY sir_numpy_c.py .
COPY sir_numpy.py .
COPY sir_sql.py .
COPY report.py .
COPY update_ages.cpp .
RUN g++ -shared -fPIC -O3 -march=native -flto -fpermissive -fopenmp update_ages.cpp -o update_ages.so
COPY model_numpy/eula.py model_numpy/ 
COPY model_sql/eula.py model_sql/ 

COPY modeled_pop.csv.gz .
#COPY cbrs.csv .
COPY fits.npy .
COPY cbrs.csv .

COPY post_proc.py .

CMD ["python", "app.py"]

