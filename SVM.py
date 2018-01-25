# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 23:58:50 2017

@author: david
"""

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
#X, y = make_classification(n_features=4, random_state=0)
X=[[1,2,3],
   [2,3,5],
   [4,3,5],
   [2,3,5],
   [2,23,15],
   [2,3,5],
   [1,3,7],
   [2,3,5],
   [12,23,15],
   [42,23,15],
   ]

y=['a','a','a','a','b','a','a','a','b','b']

clf = LinearSVC(random_state=0)
clf.fit(X, y)

LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)

print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[11,22,15]])) 
