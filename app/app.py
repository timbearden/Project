from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/about')
def index():
    return render_template('about.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/example')
def example():
    return render_template('example.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
