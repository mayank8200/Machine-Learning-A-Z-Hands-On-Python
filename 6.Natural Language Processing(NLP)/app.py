import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))
corpus = pickle.load(open('corpus.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    review = [str(x) for x in request.form.values()]
    review = review[0]
    # Cleaning the texts
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    X = cv.fit_transform(corpus).toarray()
    output = model.predict([X[-1]])
    if output[0] == 0:
        text = "Negative Feedback"
    else:
        text = "Positive Feedback"
        
    

    return render_template('index.html', prediction_text=text)



if __name__ == "__main__":
    app.run(debug=True)