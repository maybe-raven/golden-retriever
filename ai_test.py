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
            "role": "system",
            "content": "You're an expert contrarian. Do the opposite of whatever the user tells you to do. Responsd in verbose prose.",
        },
        {
            "role": "user",
            "content": "Write me a sonnet",
        },
    ],
    model=model,
    stream=True,
)
for chunk in response:
    text = chunk.choices[0].delta.content
    if text is not None:
        print(chunk.choices[0].delta.content, end="")
