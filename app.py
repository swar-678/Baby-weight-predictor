from flask import Flask,render_template,request,jsonify
import pandas as pd
import pickle

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('baby_wt.html')

@app.route('/upload',methods=['POST'])
def indata():
    data=request.form.to_dict()
    print(data)

    data_df=pd.DataFrame([data])
    data_df=data_df.astype(float)
    data_df[['weight']]*=2.20462262
    print(data_df)

    with open('model/model.pkl','rb') as f:
        mymodel=pickle.load(f)

    pred_wt=mymodel.predict(data_df)
    pred_wt=round(float(pred_wt),2)

    return render_template('baby_wt.html',pred_wt=pred_wt*28/1000)

if __name__=='__main__':
    app.run(debug=True)