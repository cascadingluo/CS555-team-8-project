from flask import Flask, render_template

app = Flask(__name__)

@app.route('/') #maps a URL path (or endpoint) to a Python function

#define the route forthe home page
def home():
    return render_template('A.V.A_interface.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)