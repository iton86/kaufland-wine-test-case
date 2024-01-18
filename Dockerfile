# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app/ /app/
COPY model/ /app/

# Install any needed packages specified in requirements_bkp.txt
RUN pip install -r app_requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Specify the command to run your script
CMD ["python", "app.py"]