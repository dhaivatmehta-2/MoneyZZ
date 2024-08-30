from flask import Flask, render_template, request
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

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        chat_session = model.start_chat(history=[{'role': 'user', 'parts': [user_input]}])
        response = chat_session.send_message(user_input)
        return render_template('index.html', user_input=user_input, response=response.text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
