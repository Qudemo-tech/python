# Use Python 3.10 base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the port your app runs on
EXPOSE 5001

# Start your app
CMD ["python", "run.py"] 