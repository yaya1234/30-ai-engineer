FROM python:3.10-alpine

WORKDIR /app

# 安装编译依赖（包含 C/C++ 编译器）
RUN apk add --no-cache build-base python3-dev linux-headers

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY app.py .
COPY titanic_rf_model.pkl .

EXPOSE 5000

CMD ["python", "app.py"]