FROM python:3.12
RUN apt-get update && apt-get install -y libgomp1
WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=300 -r requirements.txt
COPY . .
CMD ["python", "agent.py"]