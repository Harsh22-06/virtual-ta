import json

data = {"question": "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?", "image": "$(base64 -w0 project-tds-virtual-ta-q1.webp)"}
json_data = json.dumps(data)
print(json_data)