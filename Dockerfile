FROM python:3.9-slim

# We set the working directory in the container
WORKDIR /app

# We install system dependencies required for OpenCV and Rasterio
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# We copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project codebase into the container
COPY . .

# Set the default command to run the ablation study script
CMD ["python", "src/evaluation/evaluate_noise.py"]
