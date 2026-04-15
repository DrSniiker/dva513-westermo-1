FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/westermo_ann

# Default run target can be overridden in docker-compose.
CMD ["python", "unsw_nb15_ann.py"]
