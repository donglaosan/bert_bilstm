from flask import Flask, request, render_template
import main

import os
from flask import Flask

app = Flask(__name__)
app.template_folder = os.path.abspath('templates')  # 设置模板文件夹路径


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        result = main.main(text)  # 使用你的模块进行文本情感预测
        return render_template('result.html', result=result)
    else:
        return 'Unsupported method'

if __name__ == '__main__':
    app.run(debug=True)
