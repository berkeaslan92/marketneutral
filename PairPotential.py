import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import OPTICS
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



class PairClusters:
    """
    Class to work on the (III.) Pairs Selection Framework.
    The aim of this class is to find potential pairs for the Pair Trading Strategy.
    If the investor limits itself for stocks that are traded in the same sector,
    it might limit the profits. Our objective is therefore to find pairs in a large trading universe.

    The first algorithm that we will use is called OPTICS (similar to DBSCAN).
    OPTICS is an Unsupervised Learning Clustering algorithm, hence:
    - No need to select the number of clusters from the beginning.
    - No need to group the securities in advance.
    - No preliminary assumptions for the clusters' composition

    Pair selection criteria steps:
    1. Pair is eligible for trading if two securities are cointegrated -> Engle-Granger test.
    """

    def __init__(self, price_data: pd.DataFrame, returns_data: pd.DataFrame):
        self.__price_data = price_data
        self.__returns_data = returns_data
        self.__clusters = None
        self.__n_clusters = None
        self.__pca_data = None
        self.__potential_pairs = []

    @staticmethod
    def standard_scaler(data):
        """
        Method
        :param data: Prices or returns data.
        :return: Standard scaled version of the data.
        """
        standard_scaler = StandardScaler()
        return standard_scaler.fit_transform(data)

    def pca(self, n_components: int = 15):
        """
        Method to apply PCA for price or return matrix.
        :param: n_components: Number of components for the PCA calculator.
        :return: None.
        """
        pca = PCA(n_components=n_components)
        pca.fit(self.standard_scaler(self.__returns_data))
        pca_data = pd.DataFrame(pca.components_.T)

        # Change the name of the data
        pca_data.index = self.__returns_data.columns
        pca_data = pca_data.add_prefix('Feature_')

        # Set the class attribute to the PCA data
        self.__pca_data = pca_data

    def optics(self):
        """
        Method to cluster stocks using the OPTICS algorithm.
        TODO: Hyperparameter tuning needs to be applied to the optics algorithm.
        :param: data: Price or return data to work with.
        :return: clusters -> pd.DataFrame;
        :rtype: PairClusters
        """

        # Do the PCA
        self.pca()

        # Use the OPTICS algorithm from sklearn
        optics_clusters = OPTICS().fit(self.__pca_data)

        # Create a Pandas DataFrame with the stock symbols and cluster labels
        d = {'Stocks': self.__returns_data.columns.values, 'ClusterID': optics_clusters.labels_}
        clusters = pd.DataFrame(d)

        # Set the attribute
        self.__clusters = clusters

        # Find how many clusters were created
        self.__n_clusters = len(np.unique(optics_clusters.labels_))

        # Return the clusters
        return clusters

    # Method that creates potential pairs from the OPTICS cluster
    def create_potential_pairs(self, display_pairs_info=True):
        # for clusterID in self.__clusters.iloc[:, 1].unique():
        for clusterID in range(self.__n_clusters - 1):  # Select only two small clusters for test
            sub_cluster_group = self.__clusters.loc[self.__clusters['ClusterID'] == clusterID]['Stocks'].values
            self.__potential_pairs.append(set(list(combinations(sub_cluster_group, 2))))

        # Flatten the list of potential pairs from big lists of pairs
        self.__potential_pairs = [pair for pair_clusters in self.__potential_pairs for pair in pair_clusters]

        if display_pairs_info:
            self.display_clusters_info()

        return self.__potential_pairs

    def display_clusters_info(self):
        """
        Method to visualize the OPTICS clusters.
        :return: None
        """
        # print(f'There are ', {self.__n_clusters}, ' clusters selected in this run.')
        print('The clusters formed by the OPTICS after PCA: ')
        print(self.__clusters)
        print(f'The number of clusters formed: ', {self.__n_clusters})
        print('The potential pairs created before the filtering process: ')
        print(self.__potential_pairs)

    def plot_tsne(self, fig_size: tuple = (20, 12)):
        """
        Method to plot t-SNE data.
        :return: Prints tsne data.
        TODO: Change the size and alpha of the non-clustered classes
        """
        tsne = TSNE(n_components=2, random_state=0)
        tsne_obj = tsne.fit_transform(self.__pca_data)

        tsne_df = pd.DataFrame({'Comp 1': tsne_obj[:, 0],
                                'Comp 2': tsne_obj[:, 1],
                                'Stock': self.__pca_data.index,
                                'Cluster': self.__clusters['ClusterID']})

        # Split the two categories: clustered and non-clustered.
        tsne_df_clustered = tsne_df.loc[tsne_df['Cluster'] != -1]
        tsne_df_non_classified = tsne_df.loc[tsne_df['Cluster'] == -1]

        # Plot the t-SNE plot
        sns.set_style("darkgrid")

        fig, ax = plt.subplots(figsize=fig_size)

        # Plot the clustered dataset
        sns.scatterplot('Comp 1',
                        'Comp 2',
                        data=tsne_df_clustered,
                        s=40,
                        hue='Cluster',
                        ax=ax,
                        palette='Paired')

        # Plot the non_clustered dataset
        sns.scatterplot('Comp 1',
                        'Comp 2',
                        data=tsne_df_non_classified,
                        s=7,
                        alpha=0.3,
                        ax=ax,
                        color='blue')

        plt.title('t-SNE Plot')
        plt.xlabel('Comp 1')
        plt.ylabel('Comp 2')
        self._label_point(tsne_df_clustered['Comp 1'], tsne_df_clustered['Comp 2'], tsne_df_clustered['Stock'],
                          plt.gca())
        plt.show()

        pass

    @staticmethod
    def _label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'] + .02, point['y'], str(point['val']), fontsize=5)

        pass
