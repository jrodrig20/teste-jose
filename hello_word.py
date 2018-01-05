from sklearn import tree

features = [[140, 1], [130, 1],
           [150, 0], [170, 0]]
labels = [0, 0, 1, 1] # 0 é maçã e 1 é laranja

# o classificador encontra padrões nos dados de treinamento
clf = tree.DecisionTreeClassifier() # instância do classificador
clf = clf.fit(features, labels) # fit encontra padrões nos dados

# iremos utilizar para classificar uma nova fruta
print(clf.predict([[150, 0]]))