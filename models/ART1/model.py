import numpy as np

class ART1():
    """
    Adaptive Resonance Theory (ART1) Network Lippmann's 
    Algorithm in the Krose-Smagt's Version for binary
    data clustering.
    https://www.emis.de/journals/GM/vol12nr3/popovici/popovici.pdf
    """
    def __init__(self, n_features, n_clusters=2, rho=.7):
        self.rho = rho
        self.n_clusters = n_clusters
        self.W2_1 = np.ones((n_clusters, n_features))
        self.W1_2 = self.W2_1.T * (1 / (1 + n_features))


    def predict(self, X):
        n_samples, n_features = X.shape

        labels = - np.ones(n_samples)

        for i, sample in enumerate(X):
            non_matching_nodes = []
            reseted_values = []
            explored_all_map = False

            while labels[i] == -1:
                # Forward
                forward = np.dot(self.W1_2.T, sample)

                forward[non_matching_nodes] = -np.inf # Simulate disabling node
                closest_map_node = forward.argmax()

                # Backward
                expected_output = np.zeros(forward.size)
                expected_output[closest_map_node] = 1

                backward = np.dot(self.W2_1.T, expected_output)

                min_sample = np.multiply(sample, backward) # Min(sample, backward) element-wise in binary setting

                matching = np.linalg.norm(min_sample) / np.linalg.norm(sample)
                reset = matching < self.rho

                if reset:
                    non_matching_nodes.append(closest_map_node)
                    reseted_values.append((matching, closest_map_node))

                # Could expand output and add additional node instead but settling for that for now
                if len(non_matching_nodes) >= self.n_clusters:
                    reset = False
                    explored_all_map = True

                if not reset:
                    if not explored_all_map:
                        self.W2_1[closest_map_node, :] *= min_sample
                        self.W1_2[:, closest_map_node] = self.W2_1[closest_map_node, :] / (
                            .5 + np.linalg.norm(self.W2_1[closest_map_node, :])
                        )
                    else:
                        closest_map_node = max(reseted_values)[1]

                    labels[i] = closest_map_node

        return labels
