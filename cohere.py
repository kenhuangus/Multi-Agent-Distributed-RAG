import cohere
co = cohere.Client(api_key="<YOUR API KEY>")

response = co.chat(
  model="command-r-plus",
  message="Write a title for a blog post about API design. Only output the title text."
)

print(response.text) # "The Art of API Design: Crafting Elegant and Powerful Interfaces"
