FROM python:3.11.2-slim  
  
WORKDIR /app  
   
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  
  
COPY . .  
  
EXPOSE 8080  

CMD ["fastapi", "run", "practica_fastapi.py", "--port", "8080"]
