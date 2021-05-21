FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U fire cloudml-hypertune pandas==0.25.3 fastai==2.1.10 xgboost==1.2.1 
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]