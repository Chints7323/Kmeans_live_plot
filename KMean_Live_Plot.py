#TechSim+

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np

from matplotlib import style
import pandas as pd 
from sklearn.datasets import load_iris

k = 3
tolerance = 0.0001
max_iterations = 2


def Distance2Point(point1, point2):
    distance = sum((point1 - point2)**2)**0.5
    return distance


iris = load_iris()
data = iris.data[:, :2]

centroids = {}
    #initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
for i in range(k):
    centroids[i] = data[i]

classes = {}

fig = plt.figure(figsize = (15,8))
ax1 = fig.add_subplot(1,1,1)
n = 1
def animate(i):
    global n
    classes = {}
    for j in range(1):
        for classKey in range(k):
            classes[classKey] = []
            
        for dataPoint in data: 
            Distance = []
            for centroid in centroids:
                dis = Distance2Point(dataPoint, centroids[centroid])
                Distance.append(dis)
            
            minDis = min(Distance)
            minDisIndex = Distance.index(minDis)
            classes[minDisIndex].append(dataPoint)
           
        oldCentroid = dict(centroids)
        if n > 20:
            for classKey in classes:
                classData = classes[classKey]
                NewCentroid = np.mean(classData, axis = 0)
                centroids[classKey] = NewCentroid
        
        isFine = True
        for centroid in oldCentroid:
            oldCent = oldCentroid[centroid]
            curr = centroids[centroid]
            
            if np.sum((curr - oldCent)/oldCent * 100) > 0.001:
                isFine = False


    colors = 10*["red", "darkblue", "k", "b", "c"]
    ax1.clear()

    for classification in classes:
        color = colors[classification]
        for features in classes[classification]:
            ax1.scatter(features[0], features[1], color = color,s = 30)

            
    for centroid in centroids:
        color = colors[centroid]
        ax1.scatter(centroids[centroid][0], centroids[centroid][1], s = 130,
                    marker = "*", color = color)
        classData= np.array(classes[centroid])
        if n > 15:
            for cd in classData:
                point1 = [centroids[centroid][0], cd[0]]
                point2 = [centroids[centroid][1], cd[1]]
                plt.plot(point1, point2, color = color)

        
    n += 1
    plt.title("K-mean Clustering with Live Centroids Ploting")

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()




