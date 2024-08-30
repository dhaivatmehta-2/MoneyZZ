import google.generativeai as genai

# Replace 'your_api_key_here' with your actual API key
api_key = 'AIzaSyDOHtE9TTov4mmdeJ1HwcXkvWPpVkzuWxU'

genai.configure(api_key=api_key)

# Create the model
generation_config = {
  "temperature": 2,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
    # You can add initial conversation history here if needed
  ]
)

# Get user input and send the message
user_input = input("Enter your question: ")
response = chat_session.send_message(user_input)

# Print the response text
print(response.text)
