from flask import Flask, request, render_template
from summarizer.summarizer import Summarizer
from summarizer.summarizer_dev import unpickle
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

vocab = unpickle('summarizer/vocab')
count = CountVectorizer(vocabulary = vocab, stop_words='english')

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

@app.route('/run-example', methods=['POST'])
def run_example():
    text = str(request.form['example_input'])
    summarizer = Summarizer(vocab=vocab, vectorizer=count, scoring='significance')
    summarizer.fit(text)
    summary = summarizer.summary
    reduction = "Size Reduction: {}% of sentences kept".format(summarizer.reduction)
    example = [(summary, reduction)]
    return render_template('example_summary.html', data=example)

@app.route('/sign-up', methods=['POST'])
def sign_up():
    return "You just signed up"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
