import requests

url = "http://127.0.0.1:5000/upload"
files = {'file': open('uploads/upload_20250825_231127.jpg', 'rb')}
data = {'confidence_threshold': '0.8'}

response = requests.post(url, files=files, data=data)
print(response.json())
