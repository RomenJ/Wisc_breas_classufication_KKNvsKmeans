import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Cargar el conjunto de datos desde el archivo CSV
data = pd.read_csv('BreastCancer.csv')
data['diagNum'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Seleccionar características para el análisis
selected_features = data[['radius_mean', 'texture_mean']].copy()  # Usaremos solo dos características para la visualización

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(selected_features)


# Calcular los mejores y los peores valores de las diferencias

silhouette_scores=[]
for k in range (2,11):
    print("For:", k )
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(X_scaled,)
    score=silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.plot(range(2,11),silhouette_scores, marker='o' )    
plt.xlabel('Número de clusters')
plt.ylabel('Coeficiente de silhouette')
plt.title('Coeficientes k')
plt.savefig('Coeficientes_k.png')
plt.show()




# Instanciar y entrenar modelo KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled)
kmeans_labels = kmeans.predict(X_scaled)
kmeans_centers = kmeans.cluster_centers_

# Instanciar y entrenar modelo KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, data['diagNum'])

# Crear gráfico
plt.figure(figsize=(12, 5))

# Gráfico izquierdo: KMeans
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.5)
plt.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], c='black', s=200, marker='X')
plt.title('KMeans Clustering')
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')

# Gráfico derecho: KNeighborsClassifier
plt.subplot(1, 2, 2)
h = 0.02  # Step size in the mesh
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['diagNum'], cmap='viridis', s=50, edgecolors='k', alpha=0.8)
plt.title('KNeighborsClassifier')
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')

plt.tight_layout()
plt.savefig('ModelComprarisson.png')

plt.show()
