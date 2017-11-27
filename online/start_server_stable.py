# coding: utf-8
#import sys
import random
#sys.path.append('../')
from flask import Flask, request, render_template, make_response
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_moment import Moment
from wtforms.validators import Required
from wtforms import SelectField, TextAreaField, SubmitField
from model_util import baselineQA, random_sample

app = Flask(__name__)
model = baselineQA()

bootstrap = Bootstrap(app)
moment = Moment(app)
app.config['SECRET_KEY'] = 'hard to guess string'

class NnSelected(FlaskForm):
    input_question = TextAreaField('',validators=[Required()],render_kw={"rows": 2})
    submit = SubmitField('搜索')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = NnSelected()
    if form.validate_on_submit():
        print('look')
        input_question = form.input_question.data
        answers, ans2evid = model.pred(input_question)
        
        res = []
        for (ans, _) in answers:
            for i, e in enumerate(ans2evid[ans]):
                ans2evid[ans][i] = str(i+1) + '. ' + e
            
            evid = '\n\n'.join(ans2evid[ans])
            ans = ans + '      (' + str(_) + ')'
            
            res.append((ans, evid))
        return render_template('index.html', form = form, res = res )
    else:
        print('??')
        form.input_question.data = random_sample()
        return render_template('index.html', form = form)
    
@app.route('/random')
def random_text():
    return random_sample()


from werkzeug.contrib.fixers import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
    app.run(debug=0, host='0.0.0.0', port=10001)
    
