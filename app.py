# app.py
#from flask import Flask, render_template, send_file

from flask import Flask, render_template, request, url_for, Response
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
from io import BytesIO
import seaborn as sns
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('MedicalInsuranceCost.pkl', 'rb'))
train_data = pd.read_csv('Train_Data.csv')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/dataset')
def dataset():
    # Convert the sample data to a string for display
    sample_data = train_data.head().to_string()
    return render_template('dataset.html', sample_data=sample_data)

@app.route('/predict')
def predict():
    return render_template('index.html')

@app.route("/predicts", methods=['POST'])
def predicts():
    if request.method == 'POST':
        age = float(request.form['age'])

        sex = request.form['gender']
        if (sex == 'male'):
            sex_male = 1
            sex_female = 0
        else:
            sex_male = 0
            sex_female = 1

        smoker = request.form['smoker']
        if (smoker == 'yes'):
            smoker_yes = 1
            smoker_no = 0
        else:
            smoker_yes = 0
            smoker_no = 1

        bmi = float(request.form['bmi'])
        children = int(request.form['children'])

        region = request.form['region']
        if (region == 'northwest'):
            region_northwest = 1
            region_southeast = 0
            region_southwest = 0
            region_northeast = 0
        elif (region == 'southeast'):
            region_northwest = 0
            region_southeast = 1
            region_southwest = 0
            region_northeast = 0
        elif (region == 'southwest'):
            region_northwest = 0
            region_southeast = 0
            region_southwest = 1
            region_northeast = 0
        else:
            region_northwest = 0
            region_southeast = 0
            region_southwest = 0
            region_northeast = 1


        values = np.array([[age,sex_male,smoker_yes,bmi,children,region_northwest,region_southeast,region_southwest]])
        prediction = model.predict(values)
        prediction = round(prediction[0],2)


        return render_template('result.html', prediction_text='Estimate medical insurance cost is {}'.format(prediction))



@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

@app.route('/plot')
def plot():
    # Generate the histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(train_data['bmi'], kde=True, ax=ax)
    ax.set_title('Body Mass Index', fontsize=20)

    # Save it to a BytesIO object
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    # Send the image as response
    return Response(img.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)


'''from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('MedicalInsuranceCost.pkl', 'rb'))

@app.route('/')



@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])

        sex = request.form['gender']
        if (sex == 'male'):
            sex_male = 1
            sex_female = 0
        else:
            sex_male = 0
            sex_female = 1

        smoker = request.form['smoker']
        if (smoker == 'yes'):
            smoker_yes = 1
            smoker_no = 0
        else:
            smoker_yes = 0
            smoker_no = 1

        bmi = float(request.form['bmi'])
        children = int(request.form['children'])

        region = request.form['region']
        if (region == 'northwest'):
            region_northwest = 1
            region_southeast = 0
            region_southwest = 0
            region_northeast = 0
        elif (region == 'southeast'):
            region_northwest = 0
            region_southeast = 1
            region_southwest = 0
            region_northeast = 0
        elif (region == 'southwest'):
            region_northwest = 0
            region_southeast = 0
            region_southwest = 1
            region_northeast = 0
        else:
            region_northwest = 0
            region_southeast = 0
            region_southwest = 0
            region_northeast = 1


        values = np.array([[age,sex_male,smoker_yes,bmi,children,region_northwest,region_southeast,region_southwest]])
        prediction = model.predict(values)
        prediction = round(prediction[0],2)


        return render_template('result.html', prediction_text='Estimate medical insurance cost is {}'.format(prediction))





if __name__ == "__main__":
    app.run(debug=True)'''

