# Starbucks-Project

The aim of ths project was to use a months worth of data to create a strategy for sending offers to customers based on data collected by a mobile app. This includes demographics data like age and gender, as well as spend and offer history. I built a model to predict if a given person would view or complete an offer which was then used to make decisions about what offers should be sent out. All the data was provided by Starbucks as part of the Udacity Capstone Project.

A full description of the project is given in the blog post.

<h2> Files </h2>

1. Data:
    * portfolio.json - Data on offers sent out by Starbucks
    * profile.json - User data for offer recipients
    * transcript.json - Transaction and offer data for all users 
2. Starbucks_Capstone_notebook.ipynb - Jupytier notebook containg the project code
3. Starbucks_Capstone_notebook_files - Images from Jupyter
4. Starbucks Blog Post.md - Blog post about project

## Dependencies
This project uses Python 3.

The main packages used for the model and data loading are 
* numpy
* pandas
* sklearn
* imblearn

Matplotlib and Seaborn were also used for graphs and scipy.stats for statistical tools.
