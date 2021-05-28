from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy  import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Thisissupposedtobesecret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))

        return '<h1>Invalid username or password</h1>'
        #return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'

    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        return 'New user has been created!' + '<a href="/login" > click here to login </a>'
        #return '<h1>' + form.username.data + ' ' + form.email.data + ' ' + form.password.data + '</h1>'

    return render_template('signup.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.username )

@app.route('/mymodel' ,methods=['GETS','POST'])
@login_required
def XGBoost():
    if request.method == 'POST':
       list1 = []
       data1 = request.form['v1']
       data2 = request.form['v2']
       data3 = request.form['v3']
       data4 = request.form['v4']
       data5 = request.form['v5']
       data6 = request.form['v6']
       data7 = request.form['v7']
       list1 =  [data1,data2,data3,data4,data5,data6,data7]
       #list1 = list(map(float, list1))
       result = valuepredictor(list1)
    if int(result)== 1:
        prediction1 ='Given transaction is fradulent'
    else:
        prediction1 ='Given transaction is NOT fradulent'  
    return render_template('mymodel.html', name=current_user.username, userid = current_user.id , prediction1=prediction1, transactionid=data4 )

@app.route('/logistic_model' ,methods=['GETS','POST'])
@login_required
def logistic_model():
    if request.method == 'POST':
       list1 = []
       data1 = request.form['v1']
       data2 = request.form['v2']
       data3 = request.form['v3']
       data4 = request.form['v4']
       data5 = request.form['v5']
       data6 = request.form['v6']
       data7 = request.form['v7']
       list1 =  [data1,data2,data3,data4,data5,data6,data7]
       #list1 = list(map(float, list1))
       result = valuepredictor(list1)
    if int(result)== 1:
        prediction1 ='Given transaction is fradulent'
    else:
        prediction1 ='Given transaction is NOT fradulent'  
    return render_template('logistic_model.html', name=current_user.username, userid = current_user.id , prediction1=prediction1, transactionid=data4 )

def valuepredictor(list1):
    arr = np.array(list1).reshape(1,7)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(arr) 
    return result[0]


@app.route('/aboutushome')
def aboutusho():
    return render_template('aboutushome.html')

@app.route('/history')
@login_required
def history():
    return render_template('history.html',name=current_user.username)

@app.route('/logistic_model_p')
@login_required
def logistic_model_p():
    return render_template('logistic_model_p.html',name=current_user.username)

@app.route('/prediction')
@login_required
def prediction():
    return render_template('prediction.html',name=current_user.username)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/new_model_p')
@login_required
def new_model_p():
    return render_template('new_model_p.html',name=current_user.username)

@app.route('/new_model' ,methods=['GETS','POST'])
@login_required
def new_model():
    if request.method == 'POST':
       list2 = []
       data1 = request.form['v1']
       data2 = request.form['v2']
       data3 = request.form['v3']
       data4 = request.form['v4']
       data5 = request.form['v5']
       data6 = request.form['v6']
       data7 = request.form['v7']
       data8 = request.form['v8']
       data9 = request.form['v9']
       data10 = request.form['v10']
       data11 = request.form['v11']
       data12 = request.form['v12']
       data13 = request.form['v13']
       data14 = request.form['v14']
       data15 = request.form['v15']
       data16 = request.form['v16']

       list2 =  [data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16]
       #list1 = list(map(float, list1))
       answer = valuepredictor1(list2)
    if int(answer)== 1:
        prediction11 ='Given transaction is fradulent'
    else:
        prediction11 ='Given transaction is NOT fradulent'  
    return render_template('new_model.html', name=current_user.username, userid = current_user.id , prediction11=prediction11, transactionid=data4 )

def category_onehot_multcols(data,multcolumns):
    df_final = data
    final_df = data
    i=0
    for fields in multcolumns:
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:           
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1             
    df_final=pd.concat([final_df,df_final],axis=1)
    return df_final

def valuepredictor1(list2):
    arr1 = np.array(list2).reshape(1,16)
    #categorical_features = [0,1,3,4,5,6,11]
    #final_df = category_onehot_multcols(arr1, categorical_features)
    loaded_model = pickle.load(open("model1.pkl", "rb"))
    answer = loaded_model.predict(arr1) 
    return answer[0]


if __name__ == '__main__':
    app.run(debug=True)