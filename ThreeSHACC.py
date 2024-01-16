import numpy as np
import copy as cp
import scipy
import time
import random
import sklearn
import sys
import dml
from sklearn.cluster import KMeans
import warnings


class ThreeSHACC:

	def __init__(self, data, nb_clust, const_matrix, lamb, beta, k_thresh, norm_thresh, max_it, informtvt_kmeans_runs, lsi_params):

		self._data = data
		self._const_matrix = const_matrix
		self._n_instances = data.shape[0]
		self._n_features = data.shape[1]
		self._result_nb_clust = nb_clust
		self._lamb = lamb
		self._beta = beta
		self._k_thresh = k_thresh
		self._norm_thresh = norm_thresh
		self._max_it = max_it
		self._informtvt_kmeans_runs = informtvt_kmeans_runs
		self._lsi_params = lsi_params

		self._dml_S = const_matrix > 0
		self._dml_D = const_matrix < 0

		self._distance_matrix = sklearn.metrics.pairwise_distances(self._data, Y=None, metric='euclidean', n_jobs = 2)
		self._infsblt_matrix_ML = np.array((self._const_matrix + 1)/2, dtype = np.int) - np.eye(self._n_instances)
		self._infsblt_matrix_CL = np.array((self._const_matrix - 1)/2, dtype = np.int)

		self._total_ML = np.sum(self._infsblt_matrix_ML)/2
		self._total_CL = np.abs(np.sum(self._infsblt_matrix_CL))/2

		self._informtvt_matrix = np.zeros((self._n_instances, self._n_instances), dtype = np.float)

	def dml_lsi(self):

		lsi = dml.lsi.LSI(eta0 = self._lsi_params.eta0, 
			max_iter = self._lsi_params.max_iter, 
			max_proj_iter = self._lsi_params.max_proj_iter,
		 	itproj_err = self._lsi_params.itproj_err, 
		 	err = self._lsi_params.err, 
		 	supervised = False)

		lsi.fit(X = self._data, side = [self._dml_S, self._dml_D])
		transformed_data = lsi.transform(X = self._data)
		self._distance_matrix = sklearn.metrics.pairwise_distances(transformed_data, Y=None, metric='euclidean', n_jobs = 2)
		# print("Terminado computo de dml")

	def dml_weighted_lsi(self):

		lsi = dml.wlsi.WLSI(eta0 = self._lsi_params.eta0, 
			max_iter = self._lsi_params.max_iter, 
			max_proj_iter = self._lsi_params.max_proj_iter,
		 	itproj_err = self._lsi_params.itproj_err, 
		 	err = self._lsi_params.err, 
		 	weights_thresh = self._lsi_params.weights_thresh)

		lsi.fit(X = self._data, side = [self._dml_S, self._dml_D, self._informtvt_matrix])
		transformed_data = lsi.transform(X = self._data)
		self._distance_matrix = sklearn.metrics.pairwise_distances(transformed_data, Y=None, metric='euclidean', n_jobs = 2)
		# print("Terminado computo de wdml")

	def floyd_warshall(self):

		#Compute Euclidean distances
		D = sklearn.metrics.pairwise_distances(self._data, Y=None, metric='euclidean', n_jobs = 2)

		#Set distances between ml-linked instances to 0
		D[self._const_matrix == 1] = 0

		#Set distances between cl-linked instances to a value over the maximum
		max_D = np.max(D)
		D[self._const_matrix == -1] = max_D * 1.5

		V = np.shape(D)[0] 

		for k in range(V): 

			# pick all vertices as source one by one 
			for i in range(V): 

				# Pick all vertices as destination for the 
				# above picked source 
				for j in range(V): 

					# If vertex k is on the shortest path from 
					# i to j, then update the value of dist[i][j] 
					D[i,j] = min(D[i,j], D[i,k]+ D[k,j]) 

		self._distance_matrix = D

	def compute_informativity_weighted(self):
		
		partitions = np.zeros((self._informtvt_kmeans_runs, self._n_instances))
		k_values = np.linspace(int(self._n_instances/3), int(self._n_instances*(2/3)), num = self._informtvt_kmeans_runs, dtype = np.int)
		weights = np.linspace(0.5, 1.0, num = self._informtvt_kmeans_runs, dtype = np.float)

		with warnings.catch_warnings():

			warnings.simplefilter("ignore")

			for i in range(self._informtvt_kmeans_runs):

				partitions[i, :] = KMeans(n_clusters = k_values[i], max_iter = np.random.randint(10,50), init = "random", n_jobs = 2).fit(self._data).labels_

		for i in range(self._informtvt_kmeans_runs):

			for j in range(self._n_instances):

				for k in range(j, self._n_instances):

					if self._const_matrix[j,k] == 1 and partitions[i,j] != partitions[i,k]:

						self._informtvt_matrix[j,k] += 1 * weights[-i]
						self._informtvt_matrix[k,j] += 1 * weights[-i]

					elif self._const_matrix[j,k] == -1 and partitions[i,j] == partitions[i,k]:

						self._informtvt_matrix[j,k] += 1 * weights[i]
						self._informtvt_matrix[k,j] += 1 * weights[i]

		self._informtvt_matrix /= self._informtvt_kmeans_runs
		# print(np.sum(self._informtvt_matrix, axis = 1))

	def compute_informativity_non_weighted(self):
		
		partitions = np.zeros((self._informtvt_kmeans_runs, self._n_instances))
		partitions = np.zeros((self._informtvt_kmeans_runs, self._n_instances))
		k_values = np.linspace(int(self._n_instances/3), int(self._n_instances*(2/3)), num = self._informtvt_kmeans_runs, dtype = np.int)

		with warnings.catch_warnings():

			warnings.simplefilter("ignore")
			
			for i in range(self._informtvt_kmeans_runs):

				partitions[i, :] = KMeans(n_clusters = k_values[i], max_iter = np.random.randint(10,50), init = "random").fit(self._data).labels_

		for i in range(self._informtvt_kmeans_runs):

			for j in range(self._n_instances):

				for k in range(j, self._n_instances):

					if self._const_matrix[j,k] == 1 and partitions[i,j] != partitions[i,k]:

						self._informtvt_matrix[j,k] += 1
						self._informtvt_matrix[k,j] += 1

					elif self._const_matrix[j,k] == -1 and partitions[i,j] == partitions[i,k]:

						self._informtvt_matrix[j,k] += 1
						self._informtvt_matrix[k,j] += 1

		self._informtvt_matrix /= self._informtvt_kmeans_runs
		# print(np.sum(self._informtvt_matrix, axis = 1))


	def compute_Z(self):

		X = np.transpose(self._data)
		Z = np.random.randint(low = 1, high = 99, size = (self._n_instances, self._n_instances))
		np.fill_diagonal(Z, 0.0)
		Z = Z / np.sum(Z, axis = 1)
		old_Z = np.zeros((self._n_instances, self._n_instances))
		it = 0

		self._not_converged = False

		D = np.array(self._distance_matrix)**2

		while np.linalg.norm(Z - old_Z, ord = "fro") > self._norm_thresh and it < self._max_it:

			for i in range(self._n_instances):

				#Extract i-th column (instance) from matrix X (data matrix)

				x = np.asmatrix(X[:, i]).transpose()

				#Extract i-th row from matrix Z

				z_t = np.asmatrix(Z[i, :])

				#Extract i-th column from euclidean distance matrix P

				p = np.asarray(D[:, i])

				#Compute matrix X_1
				X_1 = X - (np.matmul(X, Z) - np.matmul(x, z_t))

				#Compute vector v

				v = np.asarray(np.matmul(X_1.transpose(), x) / (np.matmul(x.transpose(), x) +  self._beta)).reshape(-1)

				#Update values of i-th row of matrix Z
				old_Z = cp.deepcopy(Z)

				for k in range(self._n_instances):

					if k == i:

						Z[i,k] = 0

					elif (np.abs(v[k]) - (self._lamb*p[k])/2) > 0:

						Z[i,k] = np.sign(v[k]) * (np.abs(v[k]) - ((self._lamb*p[k])/2))

					else:

						Z[i,k] = 0


			it += 1

		if it >= self._max_it - 1:
			print("Not converged")
			self._not_converged = True

		#Apply a hard thresholding operator to Z to select the k_thresh largest entries in each column
		for i in range(self._n_instances):

			row = np.array(Z[:, i].transpose().flatten())
			below_threshold_value = np.sort(row)[::-1][self._k_thresh-1]
			Z[:,i][Z[:,i] < below_threshold_value] = 0

		# W = np.abs(Z)
		# W = (W + W.transpose())/2
		# self._SM = sklearn.preprocessing.normalize(W, norm='l2', axis = 0)
		self._SM = sklearn.preprocessing.normalize(Z, norm='l2', axis = 0)

	def update_infsblt(self, c1, c2):

		#Update ML infeasibility matrix
		self._infsblt_matrix_ML[c1, :] += self._infsblt_matrix_ML[c2, :]
		self._infsblt_matrix_ML[c1, c1] += self._infsblt_matrix_ML[c2, c2]
		self._infsblt_matrix_ML[:, c1] = self._infsblt_matrix_ML[c1, :]


		#Update CL infeasibility matrix
		self._infsblt_matrix_CL[c1, :] += self._infsblt_matrix_CL[c2, :]
		self._infsblt_matrix_CL[c1, c1] += self._infsblt_matrix_CL[c2, c2]
		self._infsblt_matrix_CL[:, c1] = self._infsblt_matrix_CL[c1, :]


	def check_consistency(self, cluster_labels):

		allocated_ML = np.sum(np.diag(self._infsblt_matrix_ML[cluster_labels, :][:, cluster_labels]))
		allocated_CL = np.abs(np.sum(np.diag(self._infsblt_matrix_CL[cluster_labels, :][:, cluster_labels])))


		nb_const_ML = (np.sum(self._infsblt_matrix_ML[cluster_labels, :][:, cluster_labels]) - allocated_ML)/2 + allocated_ML
		nb_const_CL = (np.abs(np.sum(self._infsblt_matrix_CL[cluster_labels, :][:, cluster_labels]))- allocated_CL)/2 + allocated_CL

		print("ML matrix sum: " + str(nb_const_ML))
		print("CL matrix sum: " + str(nb_const_CL))

	def get_merge_infs(self, c1, c2, cluster_labels):

		ML_copy = cp.deepcopy(self._infsblt_matrix_ML)
		CL_copy = cp.deepcopy(self._infsblt_matrix_CL)

		#Update ML infeasibility matrix
		ML_copy[c1, :] += ML_copy[c2, :]
		ML_copy[c1, c1] += ML_copy[c2, c2]
		ML_copy[:, c1] = ML_copy[c1, :]


		#Update CL infeasibility matrix
		CL_copy[c1, :] += CL_copy[c2, :]
		CL_copy[c1, c1] += CL_copy[c2, c2]
		CL_copy[:, c1] = CL_copy[c1, :]

		# print(cluster_labels)
		cluster_labels = cluster_labels[cluster_labels != c2]

		allocated_ML  = np.sum(np.diag(ML_copy[cluster_labels, :][:, cluster_labels]))

		non_allocated_CL  = self._total_CL - np.abs(np.sum(np.diag(CL_copy[cluster_labels, :][:, cluster_labels])))

		return allocated_ML + non_allocated_CL

	def get_aff(self, partition):

		#Get cluster labels and number of instances per cluster
		cluster_labels, cluster_count = np.unique(partition, return_counts = True)
		#Initialize Affinity matrix as empty
		A = np.zeros([len(cluster_labels), len(cluster_labels)])
		A[A == 0] = -np.inf

		# self.check_consistency(cluster_labels)

		max_aff = -1

		#Assign each entry of Affinity matrix
		for i in range(len(cluster_labels)):

			for j in range(i+1, len(cluster_labels)):

				#Get number of instances for clusters i and j
				card_i = cluster_count[cluster_labels == cluster_labels[i]][0]
				card_j = cluster_count[cluster_labels == cluster_labels[j]][0]

				#Get W subamtrices for cluster i and j
				cluster_i = np.where(partition == cluster_labels[i])
				cluster_j = np.where(partition == cluster_labels[j])

				W_ij = np.asmatrix(self._SM[cluster_i[0],:])[:,cluster_j[0]]
				W_ji = np.asmatrix(self._SM[cluster_j[0],:])[:,cluster_i[0]]

				#Compute first term of the affinity calculation
				v_11 = np.matrix([1/ card_i**2] * card_i)
				v_12 = np.ones((card_i, 1))

				first_term = np.matmul(v_11,W_ij)
				first_term = np.matmul(first_term, W_ji)
				first_term = np.matmul(first_term, v_12)

				#Compute second term of the affinity calculation
				v_21 = np.matrix([1/ card_j**2] * card_j)
				v_22 = np.ones((card_j, 1))

				second_term = np.matmul(v_21,W_ji)
				second_term = np.matmul(second_term, W_ij)
				second_term = np.matmul(second_term, v_22)

				infs = self.get_merge_infs(cluster_labels[i], cluster_labels[j], cluster_labels)

				A[i,j] = first_term + second_term + infs * 10000

		return A

	def run(self, distance_computation = "euclidean"):

		init_time = time.time()

		if distance_computation == "lsi":
			self.dml_lsi()
		elif distance_computation == "fw":
			self.floyd_warshall()
		elif distance_computation == "wlsi-w":			
			self.compute_informativity_weighted()			
			self.dml_weighted_lsi()
		elif distance_computation == "wlsi-nw":
			self.compute_informativity_non_weighted()
			self.dml_weighted_lsi()

		#Obtain similarity matrix using the RCPD method		
		self.compute_Z()

		#Initialize partition of the dataset. Each instances has its own cluster
		partition = np.array(range(0, self._n_instances))

		#Get the initial number of clusters
		number_clusters = self._n_instances
		it = 0		
		while number_clusters > self._result_nb_clust:

			it += 1
			#Compute affinities
			A = self.get_aff(partition)
			#Get cluster labels
			cluster_labels, cluster_count = np.unique(partition, return_counts = True)
			#Get the two clusters with the greatest affinity to merge them
			to_merge = np.unravel_index(np.argmax(A, axis=None), A.shape)
			#Merge both clusters
			partition[partition == cluster_labels[to_merge[1]]] = cluster_labels[to_merge[0]]
			self.update_infsblt(cluster_labels[to_merge[0]], cluster_labels[to_merge[1]])
			number_clusters = len(np.unique(partition, return_counts = True)[0])
			cluster_labels, cluster_count = np.unique(partition, return_counts = True)

		#Return the n_cluster-partition of the dataset
		# print(partition)
		cluster_labels, cluster_count = np.unique(partition, return_counts = True)
		# print(cluster_count)

		runtime = time.time() - init_time

		return partition, runtime, self._not_converged, self._informtvt_matrix