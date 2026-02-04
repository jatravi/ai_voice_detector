import base64

with open("sample4.mp3", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

print(b64)