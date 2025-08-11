FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y gdal-bin libgdal-dev python3-pip && \
    pip install --upgrade pip

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .
