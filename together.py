import os
import openai

system_content = "You are a travel agent. Be descriptive and helpful."
user_content = "Tell me about San Francisco"

client = openai.OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
    )
chat_completion = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ],
    temperature=0.7,
    max_tokens=1024,
)
response = chat_completion.choices[0].message.content
print("Together response:\n", response)
