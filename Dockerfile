FROM python:3.6
WORKDIR /src
COPY ./req.txt .
RUN pip install -r req.txt
COPY . .
EXPOSE 4567

