# Dockerfile
FROM python:3.11-bookworm
# Set the working directory to /app
WORKDIR /app
# Copy the requirements file
COPY requirements.txt .
# Install the dependencies
RUN pip install -r requirements.txt
# Copy the application code
COPY app.py .
# Set the environment variable for OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
# Expose the port
EXPOSE 8501
# Run the command to start the Streamlit app
CMD ["streamlit", "run", "app.py"]