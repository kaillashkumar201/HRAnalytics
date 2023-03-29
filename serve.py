import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from flask import Flask, Response, render_template, url_for
from scipy.stats import skew

# Flask Application
# This section contains code fragments that is responsible for setting up and maintaining the Flask server
application = Flask(__name__, template_folder='pages', static_folder='static')

# Data Analytics
# This section contains all functions and global analytics data
# This includes:
#   - Univariate analysis
#   - Bivariate analysis
#   - Multivariate analysis
df = pd.read_csv('res/aug_train.csv')

# STAGE: Preprocessing
# STEP(Preprocessing): Drop the `enrollee_id` column
if 'enrollee_id' in df.columns:
    df = df.drop('enrollee_id', axis=1)

# STEP(Preprocessing): Transform `city` column into type `int`
df['city'] = df['city'].apply(lambda c: int(c[5:]))

# STEP(Preprocessing): Transform Columns with `NaN` as `Unspecified`
# NOTE: Only for descriptive columns
unspecified_cols = ['gender', 'major_discipline', 'company_type']
df[unspecified_cols] = df[unspecified_cols].fillna("Unspecified")

# STAGE: Univariate analysis
univariate_analysis = dict()

# STEP(univariate): Analysing the frequency of a given attribute (Gender)
univariate_analysis['gender'] = df['gender'].value_counts().to_dict()

# STEP(univariate): Obtaining the mean, median, mode, standard deviation,
#                   variance and skewness of a given attribute
#                   (training_hours)
univariate_analysis['training_hours'] = {
    'Mean':                 df['training_hours'].mean(),
    'Median':               df['training_hours'].median(),
    'Mode':                 df['training_hours'].mode(),
    'Standard Deviation':   df['training_hours'].std(),
    'Variance':             df['training_hours'].var(),
    'Skewness':             skew(df['training_hours'])
}

fig, axis = plt.subplots()
axis.pie(df['gender'].value_counts(), labels=[
    'Male', 'Unspecified', 'Female', 'Other'])
fig.savefig('static/univariate-gender-pie.png')
fig.clf()

fig = sns.barplot(x='experience', y='training_hours', data=df).get_figure()
fig.savefig('static/bivariate-barplot.png')
fig.clf()

fig = sns.scatterplot(x='city_development_index',
                      y='training_hours', hue='target', data=df).get_figure()
fig.savefig('static/multivariate-scatterplot.png')
fig.clf()

fig = sns.heatmap(df.corr(method='pearson'), annot=True).get_figure()
fig.savefig('static/multivariate-heatmap.png')
fig.clf()

"""
@application.route('/imgs/univariate/gender-pie.png')
def univariate_gender_pie_png():
    fig, axis = plt.subplots()
    axis.pie(df['gender'].value_counts(), labels=[
             'Male', 'Unspecified', 'Female', 'Other'])
    fig.savefig('static/imgs/univariate/gender-pie.png')
    return generate_image_from_plot(fig)
"""

# STAGE: Bivariate analysis
bivariate_analysis = dict()

# STEP(bivariate): Pearson Correlation Coefficient between various
#                  attributes in the dataset
bivariate_analysis['pearson'] = df.corr(
    method='pearson').to_html(classes="data", header="Table")

# STEP(bivariate): Cross Table (Gender, Major Discipline)
bivariate_analysis['cross_table'] = pd.crosstab(
    df['gender'], df['major_discipline']).to_html(classes="data", header="Table")

@application.route('/')
def index():
    images = [
        url_for('static', filename='univariate-gender-pie.png'),
        url_for('static', filename='bivariate-barplot.png'),
        url_for('static', filename='multivariate-scatterplot.png'),
        url_for('static', filename='multivariate-heatmap.png')
    ]
    return render_template('index.html', images=images, univariate=univariate_analysis, bivariate=bivariate_analysis)


if __name__ == '__main__':
    application.run(debug=True)
