import csv
import math
import numpy as np
from unidecode import unidecode

def read_language(lan, dir):
    f = open(dir + lan, "rt", encoding="utf8").read()
    f = f.replace(',', '').replace('.', '').replace('!', '') \
        .replace(')', '').replace('(', '').replace('-', '') \
        .replace('1', '').replace('2', '').replace('3', '') \
        .replace('4', '').replace('5', '').replace('6', '') \
        .replace('7', '').replace('8', '').replace('9', '') \
        .replace('0', '').replace(';', '')

    triples = dict([])

    for w in f.split():
        w = unidecode(w.lower())
        tr = ''

        if len(w) < 3:
            if len(w) == 1:
                tr = w[0]
            elif len(w) == 2:
                tr = w[0] + w[1]
            if tr in triples:
                triples[tr] += 1
            else:
                triples[tr] = 1
        else:
            for i in range(0, len(w)):
                if i == len(w) - 1 or i == len(w) - 2:
                    continue
                else:
                    tr = w[i] + w[i + 1] + w[i + 2]

                if tr in triples:
                    triples[tr] += 1
                else:
                    triples[tr] = 1

    return triples

def read_files(files, languages, dir):
    lan = dict([])

    for i in range(len(files)):
        lan[languages[i]] = read_language(files[i], dir)

    #print(lan)
    return lan

class HierarchicalClustering:
    def __init__(self, data):
        """Initialize the clustering"""
        self.lan = data
        # self.clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into clusterings of the type
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        self.clusters = [[name] for name in self.lan.keys()]
        self.names = []
        self.languages = [l for l in self.lan.keys()]
        self.all_clusters = [[name] for name in self.lan.keys()]
        self.distances = dict([])

    def row_distance(self, r1, r2):
        """
        Distance between two rows.
        Implement either Euclidean or Manhattan distance.
        Example call: self.row_distance("Polona", "Rajko")
        """
        sum = 0

        for i in range(len(self.data[r1])):
            sum += ((self.data[r1][i] - self.data[r2][i]) ** 2)
        return math.sqrt(sum)

    def cos_distance(self, r1, r2):
        cos = np.dot(r1, r2) / (np.linalg.norm(r1)*np.linalg.norm(r2))
        return 1 - cos

    def dict_intersection(self, l1, l2):
        #finds mutual keys
        mutual = (set(self.lan[l1].keys()) & set(self.lan[l2].keys()))

        #construct vector for each language
        #that contains frequences for the i-th key
        v1 = [0] * len(mutual)
        v2 = [0] * len(mutual)
        mutual = list(mutual)

        for i in range(len(mutual)):
            v1[i] = self.lan[l1][mutual[i]]
            v2[i] = self.lan[l2][mutual[i]]

        return v1, v2

    def diff_all_languages(self):
        # get the distances between all languages
        for i in self.languages:
            for j in self.languages:
                v1, v2 = self.dict_intersection(i, j)
                self.distances[i, j] = self.cos_distance(v1, v2)


    def collect_countries(self, c, countries):
        if type(c) is str:
            return

        for i in c:
            if type(i) is str:
                countries.append(i)
            else:
                self.collect_countries(i, countries)

    def spaces(self, n):
        s = ""
        for i in range(n):
            s += '    '
        return s

    def rek(self, lista, n):

        if type(lista[0]) is str:
            s = self.spaces(n)
            print(s, '----', lista[0])
            self.names.append(lista[0])
            return

        if len(lista) == 2:
            self.rek(lista[0], n + 1)
            s = self.spaces(n)
            print(s, '----|')
            self.rek(lista[1], n + 1)

    def cluster_distance(self, c1, c2):
        """
        Compute distance between two clusters.
        Implement either single, complete, or average linkage.
        Example call: self.cluster_distance(
            [[["Albert"], ["Branka"]], ["Cene"]],
            [["Nika"], ["Polona"]])
        """
        sum = 0

        c1_all = []
        self.collect_countries(c1, c1_all)
        c2_all = []
        self.collect_countries(c2, c2_all)

        for pi in c1_all:
            for pj in c2_all:
                sum += self.distances[pi, pj]

        distance = sum / (len(c1_all) * len(c2_all))
        return distance

    def closest_clusters(self):
        """
        Find a pair of closest clusters and returns the pair of clusters and
        their distance.

        Example call: self.closest_clusters(self.clusters)
        """
        min = self.cluster_distance(self.clusters[0], self.clusters[1])
        c1 = self.clusters[0]
        c2 = self.clusters[1]

        for i in range(len(self.clusters)):
            for j in range(len(self.clusters)):
                if i < j:
                    distance = self.cluster_distance(self.clusters[i], self.clusters[j])
                    if min > distance:
                        min = distance
                        c1 = self.clusters[i]
                        c2 = self.clusters[j]

        return c1, c2, min

    def run(self):
        """
        Given the data in self.data, performs hierarchical clustering.
        Can use a while loop, iteratively modify self.clusters and store
        information on which clusters were merged and what was the distance.
        Store this later information into a suitable structure to be used
        for plotting of the hierarchical clustering.
        """

        #at the beginning, every country represents a cluster
        no_clusters = len(self.clusters)

        while no_clusters > 1:

            #find the closest clusters and store information
            x, y, d = self.closest_clusters()

            #merge them as one
            #delete the separate ones
            self.clusters.remove(x)
            self.clusters.remove(y)
            self.clusters.append([x, y])
            #print(x, y, d)
            if d <= 0.43:
                self.all_clusters.remove(x)
                self.all_clusters.remove(y)
                self.all_clusters.append([x, y])

            no_clusters -= 1

        #to je zarad tega ker doda se en [] na koncu pa to moti plot_tree() funkcije
        self.clusters = self.clusters[0]
        for i in self.all_clusters:
            print(i)

    def plot_tree(self):
        """
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """
        self.rek(self.clusters, 0)
        #print(self.names)

if __name__ == "__main__":
    files = ['slv.txt', 'src5.txt', 'mkj.txt', 'blg.txt', 'rus.txt', 'ruw.txt', 'spn.txt', 'por.txt', 'gln.txt',
                 'frn.txt', 'itn.txt', 'ger.txt', 'eng.txt', 'dut.txt', 'fri.txt', 'nrn.txt', 'dns.txt', 'swd.txt',
                 'grk.txt', 'aln.txt']
    languages = ['slv', 'src5', 'mkj', 'blg', 'rus', 'ruw', 'spn', 'por', 'gln', 'frn', 'itn', 'ger', 'eng', 'dut',
                     'fri', 'nrn', 'dns', 'swd', 'grk', 'aln']

    hc = HierarchicalClustering(read_files(files, languages, 'ready\\'))
    hc.diff_all_languages()
    hc.run()
    hc.plot_tree()