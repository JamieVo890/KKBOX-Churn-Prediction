FROM python:3.11.7-slim


WORKDIR /app


COPY requirements.txt .
COPY setup.py .


RUN pip install --no-cache-dir -r requirements.txt

COPY src/components/__init__.py ./src/components/
COPY src/components/data_ingestion.py ./src/components/
COPY src/components/data_transform.py ./src/components/
COPY src/components/model_trainer.py ./src/components/

COPY src/entity/__init__.py ./src/entity/
COPY src/entity/config_entity.py ./src/entity/

COPY src/pipeline/__init__.py ./src/pipeline/
COPY src/pipeline/train_pipeline.py ./src/pipeline/

COPY src/__init__.py src/
COPY src/exception.py src/
COPY src/logger.py src/
COPY src/utils.py src/

# Set the default command to run your application
CMD ["python", "src/pipeline/train_pipeline.py"]