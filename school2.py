import csv
import random
import plotly_express as px
import pandas as pd
import statistics as st
import numpy as np
import plotly.graph_objects as go
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
import seaborn as sns

# df = pd.read_csv("heightweight.csv")
# heightList = df["Height"].to_list()
# weightList = df["Weight"].to_list()

df2 = pd.read_csv("school2.csv")
greList = df2["TOEFL Score"]
chanceList = df2["GRE Score"]
admissionList = df2["Chanceofadmit"]

score_train, score_test, result_train, result_test = train_test_split(greList, admissionList, test_size = 0.25, random_state=0)

x = np.reshape(score_train.ravel(), (len(score_train), 1))
y = np.reshape(result_train.ravel(), (len(result_train), 1))

classifier = LogisticRegression(random_state=0)
classifier.fit(x, y.ravel())

x_test = np.reshape(score_test.ravel(), (len(score_test), 1))
y_test = np.reshape(result_test.ravel(), (len(result_test), 1))

result_pred = classifier.predict(x_test)

print("Accuracy Score: ",accuracy_score(result_test, result_pred))

# heightArray = np.array(heightList)
# weightArray = np.array(weightList)

greArray = np.array(greList)
chanceArray = np.array(chanceList)

# b, a = np.polyfit(heightArray, weightArray, 1)
b2, a2 = np.polyfit(greArray, chanceArray, 1)

# with open("heightweight.csv", newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)

# a = 0
# b = 1

y = []

y2 = []

# for x in heightArray:
#     y_value = b*x + a
#     y.append(y_value)

for x in greArray:
    y2_value = b2*x + a2
    y2.append(y2_value)

# x2 = float(input("TOEFL Score: "))
# y3_value = b2*x2 + a2
# print(f"Chance of Admission With Score of {x2} is {y3_value * 100}%")

# fig = px.scatter(df, x = heightArray, y = weightArray, title = "Height and Weight")
# fig.update_layout(shapes = [
#     dict(
#         type = 'line',
#         y0 = min(weightArray),
#         y1 = max(weightArray),
#         x0 = min(heightArray),
#         x1 = max(heightArray)
#     )
#                                                                                                                                                                                                                                                                 ]
# )

# fig.show()

fig2 = px.scatter(df2, x = greArray, y = chanceArray, title = "TOEFL Score and Chances of Admission")
fig2.update_layout(shapes = [
    dict(
        type = 'line',
        y0 = min(y2),
        y1 = max(y2),
        x0 = min(greArray),
        x1 = max(greArray)
    )
])

fig2.show()

colors = []

for data in admissionList:
    if data == 1:
        colors.append("green")
    else:
        colors.append("red")

fig = go.Figure(go.Scatter(x = greList, y = chanceList, mode = 'markers', marker = dict(color=colors)))
fig.show()
