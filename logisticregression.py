import numpy
import csv
from matplotlib import pyplot as plt
import math
Pclass = []
Sex = []
SibSp = []
rescue = []
with open('titanictrain.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for inputdata in reader:
        Pclass.append(inputdata['Pclass'])
        if 'female' == inputdata['Sex']:
            Sex.append("0")
        else:
            Sex.append("1")
        SibSp.append(inputdata['SibSp'])
        rescue.append(inputdata['Survived'])
pclass = numpy.asarray(Pclass,dtype=int)
sex = numpy.asarray(Sex,dtype=int)
sibsp = numpy.asarray(SibSp,dtype=int)
survived = numpy.asarray(rescue,dtype=int)
theta1, theta2, theta3, theta4 = 0, 0, 0, 0
learning_rate, iterations = 1, 1000
len = pclass.size
for i in range(iterations):
    ans1, ans2, ans3, ans4 = 0, 0, 0, 0
    for i in range(len):
        highpothetic = (1 / (1 + math.exp(-1 * (theta4 * pclass[i] + theta3 * sex[i] + theta2 * sibsp[i] + theta1))))
        ans1 += (highpothetic - survived[i]) * 1
        ans2 += (highpothetic - survived[i]) * sibsp[i]
        ans3 += (highpothetic - survived[i]) * sex[i]
        ans4 += (highpothetic - survived[i]) * pclass[i]
    theta1 -= (learning_rate / len) * ans1
    theta2 -= (learning_rate / len) * ans2
    theta3 -= (learning_rate / len) * ans3
    theta4 -= (learning_rate / len) * ans4
i = 0
g = []
k = 0
with open('titanicgender_submission_eample.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for inputdata in reader:
        g.append(inputdata['Survived']);
        
g = numpy.asarray(g,dtype=int)
with open('titanictest.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for inputdata in reader:
        k+=1
        if inputdata['Sex'] == 'female':
            gender = 0
        else:
            gender = 1
        pcla = (int)(inputdata['Pclass'])
        tt = (int)(inputdata['SibSp'])
        ans = (theta4 * pcla + theta3 * gender + theta2 * tt + theta1)
        x = (1 / (1 + math.exp(-1 * ans)))
        if x > 0.5:
            x = 1
        else:
            x = 0
        if x != g[k-1]:
            i += 1
print(i)
print(k)
