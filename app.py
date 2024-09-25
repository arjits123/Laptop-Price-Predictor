# Importing the necessary libraries
from flask import Flask, render_template, request # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from src.pipelines.prediction import CustomData, PredictionPipeline

#creat flask application
application = Flask(__name__)
app = application

#upload dataset 
laptop = pd.read_csv('artifacts/cleaned_data.csv')

company = sorted(laptop['Company'].unique())
type_name = sorted(laptop['TypeName'].unique())
Ram = sorted(laptop['Ram'].unique())
cpu = sorted(laptop['Cpu brand'].unique())
hdd = sorted(laptop['HDD'].unique())
ssd = sorted(laptop['SSD'].unique())
gpu = sorted(laptop['Gpu brand'].unique())
os = sorted(laptop['os'].unique())
screenResolution = ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']

@app.route('/')
def index():
    return render_template('index.html',
                            companies = company, 
                            typeNames = type_name,
                            rams = Ram,
                            screen_res = screenResolution,
                            cpus = cpu,
                            hdds = hdd,
                            ssds = ssd,
                            gpus = gpu,
                            operating_system = os)

@app.route('/', methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
    # get the data from website
        data = CustomData(
            company = request.form.get('company'), 
            type_name = request.form.get('type_name'), 
            Ram = request.form.get('ram'),
            weight = request.form.get('weight'), 
            touchScreen = request.form.get('touchScreen'), 
            ips = request.form.get('ips'), 
            screenSize = request.form.get('screenSize'), 
            screenResolution = request.form.get('screenResolution'), 
            cpu_brand = request.form.get('cpu_brand'), 
            HDD = request.form.get('hdd'), 
            SSD = request.form.get('ssd'),
            GPU_brand = request.form.get('gpu_brand'), 
            os = request.form.get('os')
        )

        # create the data frame
        prediction_df = data.get_data_as_df()
        print(prediction_df)

        #initialise the pipeline
        pipe = PredictionPipeline()
        results = pipe.predict(features = prediction_df)
        # final_result = np.round(int(np.exp(results)),2)
        final_result = int(np.exp(results.item()))
        print(final_result)
        return render_template('index.html', result = final_result) 

if __name__ =='__main__':
    app.run(debug = True, port = 5001)