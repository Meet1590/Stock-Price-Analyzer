# base image - on top of which the whole application will be build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Sets working dir in vm, where all the subsequent commands will get executed
# will create directory if it doesn't exist
WORKDIR /app

# only copy requirements.txt
# Command details COPY <-- Image folder --> <--host folder--> 
# we often use it to copy dependencies, configs & Source
COPY requirements.txt requirements.txt

# Runs command in container at build time
# Common for Dependencis and config set-up
RUN pip install --no-cache-dir -r requirements.txt

#Copy source - often changes
COPY . .

# I used copy twice becasue I would like to reduce load of re-running
# pip install if nothing is chnaged in requirements.txt


# This commands will be executed once the container is started
CMD ["streamlit", "run", "app.py", "--server.port=8501" ,"--server.address=0.0.0.0"]

# Specifies that container will listen on a mentioned port
# main purpose: Documentation
EXPOSE 8501

# Port is actually published by "-p" flag

# Sets environment variables
# ENV

# defines build time variables - only works during build stage
# ARG

# Entrypoint: defines commands always runs in the container
# it has a higher priority than cmd, also it allows container to run as an executable

# USER: allows to specify different user than root

# Useful for images which can be used as a base image for other images
# ONBUILD: adds trigger instruction to an image, which will execute image when used as a base image in another build

# LABEL: Adds meta-data to the image

# STOPSIGNAL: system call to stop the container

# HEALTHCHECK: to check if the container is running as expected or not
# diff params can be specified to check helth, mark container as unhealthy, which is uesful in container orchestration