# Use a newer base image with Python 3.10 or higher
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose port and run the app
EXPOSE 5000
CMD ["python", "run.py"]
