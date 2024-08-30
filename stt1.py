from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the speech-to-text result from the form
        user_input = request.form.get('user_input')
        return render_template('stt1html.html', user_input=user_input)

    return render_template('stt1html.html')

if __name__ == '__main__':
    app.run(debug=True)
