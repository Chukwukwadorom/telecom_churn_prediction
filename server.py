from flask import Flask, request, jsonify, render_template, url_for, redirect
from flask_bootstrap import Bootstrap
import tensorflow as tf
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SelectField
from wtforms.validators import DataRequired
from dotenv import load_dotenv
import os
from helper import feature_engineer, scale_and_transform, create_model
import pandas as pd



# from nbimporter import Notebook

# with Notebook():
#     from Tel_churn import feature_engineer


load_dotenv()



secret_key = os.getenv('SECRET_KEY')

app = Flask(__name__)

app.config['SECRET_KEY'] = secret_key
Bootstrap(app)
       
     


choice_Network_sub = [('2G', "2G") ,('3G', "3G") ,('Other', "Other"), ('unknown',"unknown")]
Most_Loved_Competitor_network = [('Uxaa',"Uxaa"), ('Weematel',"Weematel"), ('unknown',"unknown"), ('Zintel',"Zintel"), ('Mango',"Mango"), ('ToCall',"ToCall"), ('PQza',"PQza")]

class DataForm(FlaskForm):
    network_age = FloatField('Network age', validators=[DataRequired()])
    Customer_tenure_in_month = FloatField('Customer tenure in month', validators=[DataRequired()])
    Total_Spend_in_Months_1_and_2_of_2017 = FloatField('Totals Spend in Months 1 and 2', validators=[DataRequired()])
    Total_SMS_Spend = FloatField('Total SMS Spend', validators=[DataRequired()])
    Total_Data_Spend = FloatField('Total Data Spend ', validators=[DataRequired()])
    Total_Data_Consumption = FloatField('Total Data Consumption', validators=[DataRequired()])
    Total_Unique_Calls = FloatField('Total Unique Calls', validators=[DataRequired()])
    Total_Onnet_spend_ = FloatField('Total Onnet spend', validators=[DataRequired()])
    Total_Offnet_spend = FloatField('Total Offnet spend', validators=[DataRequired()])
    Total_Call_centre_complaint_calls = FloatField('Total Call centre complaint calls', validators=[DataRequired()])
    Network_type_subscription_in_Month_1 = SelectField('Network type subscription in Month 1', choices=choice_Network_sub)
    Network_type_subscription_in_Month_2 = SelectField('Network type subscription in Month 2', choices=choice_Network_sub)
    Most_Loved_Competitor_network_in_Month_1 = SelectField('Most Loved Competitor network in Month 1', choices=Most_Loved_Competitor_network)
    Most_Loved_Competitor_network_in_Month_2 = SelectField('Most Loved Competitor network in Month 2', choices=Most_Loved_Competitor_network)




@app.route('/', methods=["POST", "GET"])
def index():
    form = DataForm() 
    if request.method =="POST":

        data = {**form.data}
        data.pop("csrf_token", None)
        df = pd.DataFrame([data])
        df = feature_engineer(df)
        df, categorical_features, numerical_features  = scale_and_transform(df)
      
        
        model = create_model(df, categorical_features, numerical_features)
        model.load_weights("model_0.7524.h5")
        
        predictions = model.predict([df[numerical_features].values] + [df[col].values for col in categorical_features])
       

        
     
        return redirect(url_for("predict", predictions=predictions))
    print("rendering form")
    return render_template("index.html", form = form)


# @app.route("/predict_batch", methods = ["POST"])
# def predict_batch():
#     pass


@app.route("/predict", methods =["POST","GET"])
def predict():
    # if request.method =="POST":
    # and form.validate_on_submit():
        #convert to dataframe

    if request.method == "POST":
        data = request.get_json()

        df = pd.DataFrame(data)
        df = feature_engineer(df)
        df, categorical_features, numerical_features  = scale_and_transform(df)
      
        
        model = create_model(df, categorical_features, numerical_features)
        model.load_weights("model_0.7524.h5")
        
        pred = model.predict([df[numerical_features].values] + [df[col].values for col in categorical_features])
        
        return jsonify(pred.tolist()), 200


    pred = request.args.get("predictions")
  

    return pred




if __name__ == '__main__':
    app.run(debug=True)
