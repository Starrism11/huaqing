# 创建模块 mymodule.py
# mymodule.py
def say_hello():
    return "Hello from module!"

# 导入第三方模块
import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # 200
