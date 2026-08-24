import requests
import json
import os

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    print("ERROR: No API key found.")
    exit(1)

models = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free"
]

for model in models:
    print(f"\nTesting: {model}")
    try:
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a JSON API. Only output valid JSON."},
                    {"role": "user", "content": "Reply with this JSON: {\"status\": \"ok\", \"model\": \"working\"}"}
                ],
                "response_format": {"type": "json_object"}
            },
            timeout=30
        )
        print(f"  HTTP Status: {res.status_code}")
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"]
            print(f"  Response: {content}")
            print(f"  [OK] WORKING")
        else:
            print(f"  Error body: {res.text[:300]}")
            print(f"  [FAIL]")
    except Exception as e:
        print(f"  Exception: {e}")
        print(f"  [FAIL]")
