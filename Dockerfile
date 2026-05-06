# Imagen base ligera y compatible
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

#Enviar archivos innecesarios y mejorar logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

#Instalar dependencias del sistema necesarias para OpenCV y ML
RUN apt-get update && apt-get install -y \
	python3.10 \
	python3-pip \ 
	build-essential \
	ffmpeg \
	libgl1 \
	libglib2.0-0 \
	git \
	cuda-nvcc-12-2 \
	libcufft-dev-12-2 \
	&& rm -rf /var/lib/apt/lists/*

#Alias para usar "python" directamente
RUN ln -s /usr/bin/python3.10 /usr/bin/python

#Directorio de trabajo
WORKDIR /app

#Copiar requeriments primero (mejora cache del build)
COPY requirements.txt .

#Actualizar pip e instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

#Copiar codigo del proyecto
COPY src/ ./src
### COPY scripts/ ./scripts

# Copiar main.py
COPY main.py .

#Crear carpetas necesarias 
RUN mkdir -p /app/data /app/output/models /app/output/logs /app/models

#Comando por defecto
CMD ["python", "main.py"]
