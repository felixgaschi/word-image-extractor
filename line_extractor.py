from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, color, io, img_as_float
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import copy


class PersistenceInterval:
    """
    Class that keeps a set of the filtration in memory with the following attributes:
    - target: the corresponding birth point in the filtered topological point (in our case the index of the local minimum where the set is created)
    - birth: the value of homology at birth
    - death: the value of homology at death
    - children: the set of PersistenceInterval that are nested in this
    - points: the set of points contained by the set
    """

    def __init__(self, target, birth, death=None, children=[]):
        self.target = target
        self.birth = birth
        self.death = death
        self.children = children
        self.points = [target]
    
    def appendChild(self, child):
        """
        Adds a persistence interval as child to this along with its points
        """
        self.points += child.points.copy()
        self.children.append(child)

    def appendPoint(self, point):
        """
        Adds a poit to the set
        """
        self.points.append(point)
    
    def getPosition(self):
        """
        Returns the position of the set
        """
        return self.target, min(self.points), max(self.points)
    
    def getCoord(self):
        """
        Gets the coordinates of the set on persistence diagram
        """
        return (self.birth, self.death)
    
    def getRelevance(self):
        """
        Gets the coordinates of the set on persistence diagram
        """
        assert self.death is not None, "Trying to get relevance of set with missing death."
        return self.death - self.birth


class LineExtractor:
    """
    Estimator for extracting lines with persistence topology with the following attributes:
    - infty: the value to give to a persistence interval which never dies (in our case, more than the double of 255 to be sure)
    - epsilon: threshold for the persistence diagram
    """

    def __init__(self, infty = 600, epsilon = 5):
        self.infty = infty
        self.epsilon = epsilon
    
    def get_filtration(self, x):
        """
        Computes the filtration of the function which values are stored in x
        Return a single persistence interval which is the father of all the others
        """
        n = x.shape[0]
        s = sorted([(i, x[i]) for i in range(n)], key=lambda x: x[1])
        selected = [False for i in range(n)]
        
        sets = {}
        ancestor = {i: i for i in range(n)}
        i = 0
        while False in selected:
            newpoint = s[i]
            j = s[i][0]
            val = s[i][1]

            selected[j] = True

            if j == 0 and selected[1]:
                ancestor[0] = ancestor[1]
                sets[ancestor[1]].appendPoint(0)
            elif j == 0:
                sets[0] = PersistenceInterval(0, val)
            elif j == n - 1 and selected[n - 2]:
                ancestor[n - 1] = ancestor[n - 2]
                sets[ancestor[n - 2]].appendPoint(n -1)
            elif j == n - 1:
                sets[n - 1] = PersistenceInterval(n - 1, val)
            elif selected[j - 1] and selected[j + 1]:
                i_a = ancestor[j - 1]
                i_b = ancestor[j + 1]
                a = x[i_a]
                b = x[i_b]
                if a < b:
                    ancestor[j] = i_a
                    for key in range(n):
                        if ancestor[key] == i_b:
                            ancestor[key] = i_a
                    sets[i_b].death = val
                    sets[i_b].appendPoint(j)
                    sets[i_a].appendChild(sets[i_b])
                    sets[i_a].appendPoint(j)
                else:
                    ancestor[j] = i_b
                    for key in range(n):
                        if ancestor[key] == i_a:
                            ancestor[key] = i_b
                    sets[i_a].death = val
                    sets[i_a].appendPoint(j)
                    sets[i_b].appendChild(sets[i_a])
                    sets[i_b].appendPoint(j)
            elif selected[j - 1]:
                ancestor[j] = ancestor[j - 1]
                sets[ancestor[j - 1]].appendPoint(j)
            elif selected[j + 1]:
                ancestor[j] = ancestor[j + 1]
                sets[ancestor[j + 1]].appendPoint(j)
            else:
                sets[j] = PersistenceInterval(j, val)

            i += 1

        sets[s[0][0]].death = self.infty

        setList = sorted([sets[i] for i in sets.keys()], key=lambda x:x.getRelevance(), reverse=True)

        self.sets = setList
        return setList

    def get_segments(self, sets=None):
        """
        gets the list of segments from a given list of persistence interval and
        a threshold epsilon (self.epsilon)
        """
        if sets is None:
            if self.sets is not None:
                sets = self.sets
            else:
                raise ValueError("sets and self.sets attributes are None, \
                    you need either to pass an origin argument to get_segments or \
                    to use get_filtration method before")
        segments = []
        for s in sets:
            if self.epsilon <= s.getRelevance():
                t, a, b = s.getPosition()
                for i, seg in enumerate(segments):
                    tp, ap, bp = seg
                    if t >= tp and bp > a:
                        bp = a
                    elif t <= tp and ap < b:
                        ap = b
                    segments[i] = (tp, ap, bp)
                segments.append((t, a, b))
        return segments

    def get_coordinates(self, sets=None):
        """
        gets the coordinates of all persistence interval on the 
        persistence diagram
        """
        if sets is None:
            if self.sets is not None:
                sets = self.sets
            else:
                raise ValueError("sets and self.sets attributes are None, \
                    you need either to pass an origin argument to get_coordinates or \
                    to use get_filtration method before")
        coords = []
        for s in sets:
            coords.append(s.getCoord())
        return coords

    def show_diagram(self, sets=None):
        """
        Plots the persistence diagram with threshold
        """
        if sets is None:
            if self.sets is not None:
                sets = self.sets
            else:
                raise ValueError("sets and self.sets attributes are None, \
                    you need either to pass an origin argument to show_diagram or \
                    to use get_filtration method before")
        coords = self.get_coordinates(sets=sets)
        x = [c[0] for c in coords]
        y = [c[1] for c in coords]
        plt.scatter(x, y)
        plt.plot(np.arange(0, max(x)), self.epsilon + np.arange(0, max(x)))
        plt.plot(np.arange(0, max(x)), np.arange(0,max(x)), c="black")

    
    def transform(self, imgList):
        """
        Gets the list of segmented line from a given image
        - imgList: list of images on which to perform the extraction
        """
        res = []
        for img in tqdm(imgList):
            y_mean = np.mean(img, axis=1)
            self.get_filtration(y_mean)
            seg = self.get_segments()
            seg = sorted(seg, key=lambda x:x[0])
            res.append(seg)
        return res
