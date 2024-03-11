FROM python:3.9-slim

WORKDIR /app

COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update
RUN apt install -y wget

COPY service/. .

COPY src/settings.py .
COPY src/tlc.py .
COPY src/sir_numpy_c.py .
COPY src/sir_numpy.py .
COPY src/sir_sql.py .
COPY src/report.py .
COPY src/update_ages.so .
COPY src/model_numpy/eula.py model_numpy/eula.py
COPY src/model_sql/eula.py model_sql/eula.py
RUN wget https://packages.idmod.org:443/artifactory/idm-data/laser/modeled_pop.csv.gz
COPY src/viz/plot_spatial.py .

COPY service/fits.npy .

CMD ["python", "app.py"]

