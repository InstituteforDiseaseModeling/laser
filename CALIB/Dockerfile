# Use Rocky Linux as base
FROM rockylinux:9

# Install Python 3.12 and dependencies
RUN python3 -m ensurepip && \
    python3 -m pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script into the container
COPY laser.py .
#COPY params.json .

# Run the script when the container starts
CMD ["python3", "laser.py", "-p", "shared/params.json", "-o", "shared/simulation_results.csv"]
