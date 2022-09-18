# Fireadora


![image](https://user-images.githubusercontent.com/113373405/190914702-9fb4f88d-efc2-4a16-8574-4efc73d1f3ba.png)



**Motivation**

All over the world, forest fires causes severe damages to valuable natural environment and loss of human lives. Forest fires are a major environmental issue, creating economical and ecological damage
while endangering human lives. The ability of accurately predicting the area that may be involved in a forest fire event may help in optimizing fire management efforts. Fast detection is a key element
for controlling such phenomenon. The fire statistics showed that the number of forest fire are caused by natural changes like oxygen level, temperature and humidity of the
area. Based on the analyzed fundamentals of the weather conditions together with the city related data in relation to the fire occurrence, we have developed a system to predict forest fire danger

**Use**

Machine learning models train on data. So, we take real life examples of forest fires that took place and collect the data prior to the fire taking place, which
is publicly available. We have the inputs as oxygen, humidity, temperature and the output as 0 or 1 based on whether or not a fire took place. On creating
a large enough dataset, we can create a trained machine learning model which can successfully predict the probability of a fire taking place in an area given
the 3 parameters. Government can in that sense take necessary precautions for areas which high probability of a fire breaking out.

**Framework**

When this concept is applied development, we can create a web application that simply takes 3 inputs from the user to get the forest fire probability.
Streamlit: Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data
science. In just a few minutes you can build and deploy powerful data apps.

**Algorithms**

This particular problem comes under the category of Supervised Learning. We train our machine learning mode using the following 3 learning models and compare the accuracies:
1. Linear Regression
2. Logistic Regression
3. Support Vector Machine 

![image](https://user-images.githubusercontent.com/113373405/190914722-2a05318c-f59c-4350-9196-2d1c0f71047b.png)

These are the following accuracies of the models, we can see that logistic regression has the highest accuracy. Thus, the model is built on it



**Requirements**

Python 3.6 or greater
Streamlit

Libraries:

SKlearn

Seaborn

Pandas

Matplotlib

