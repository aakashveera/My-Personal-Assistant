# Use the specified image as the base
FROM python:3.10-slim-bullseye as release

COPY ./src/llm_api/requirements.txt ./

# Install Poetry using pip and clear cache
RUN pip install --no-cache-dir -r ./requirements.txt

# Set the working directory
WORKDIR /app

# Copy the poetry lock file and pyproject.toml file to install dependencies
COPY ./src /app/src
COPY ./run_llm_api_app.sh /app
RUN mkdir /app/logs/

# Give execution permission to your shell script
RUN chmod +x /app/run_llm_api_app.sh

# Run your shell script
CMD ["./run_llm_api_app.sh"]
