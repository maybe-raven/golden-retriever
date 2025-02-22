import os
from openai import OpenAI

model = os.environ.get("GR_MODEL", "mistral-nemo-instruct-2407")

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "..."),
)
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Write me a sonnet",
        }
    ],
    model=model,
)
print(response.choices[0].message.content)
