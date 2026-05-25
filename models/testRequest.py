import requests

url  = "http://localhost:5000/predict"
img  = "dataset/Testing/glioma/Te-gl_10.jpg"

with open(img, "rb") as f:
    resp = requests.post(url, files={"file": f})

data = resp.json()  
print(f"Prediction : {data['prediction']}")
print(f"Confidence : {data['confidence']}%")
print(f"All probs  : {data['probabilities']}")
