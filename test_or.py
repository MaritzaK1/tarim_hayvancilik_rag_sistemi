from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
try:
    response = client.chat.completions.create(
        model="openrouter/auto",
        messages=[{"role": "user", "content": "Hi"}],
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(e)
