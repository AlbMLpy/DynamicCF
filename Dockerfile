FROM python:3.10-slim

WORKDIR /dcf

COPY environment/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "pytest", "--disable-warnings" ]
