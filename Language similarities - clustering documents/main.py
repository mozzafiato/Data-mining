import random
import numpy as np
from unidecode import unidecode
import matplotlib.pyplot as plt
import operator

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

class KMedoids:
    def __init__(self, lan):
        self.lan = lan
        self.languages = [l for l in self.lan.keys()]
        self.n = len(lan)
        self.distances = dict([])
        self.mapping = dict([])
        self.silh = [0] * self.n
        self.max_silhouette = -1
        self.max_index = []
        self.min_silhouette = 1
        self.min_index = []
        self.average_silh = []
        self.all_groupings = []
        self.current_iteration = 0
        self.clusters = []

    def cos_distance(self, r1, r2):
        cos = np.dot(r1, r2) / (np.linalg.norm(r1)*np.linalg.norm(r2))
        return 1 - cos

    def random_no(self, k):
        #return k different random numbers from 0 to self.n
        return random.sample(range(0, self.n), k)

    def diff_all_languages(self):
        # get the distances between all languages
        for i in self.languages:
            for j in self.languages:
                    v1, v2 = self.dict_intersection(i, j)
                    self.distances[i, j] = self.cos_distance(v1, v2)

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

    def get_cluster(self, m):
        #get languages that correspond to medoid m
        cluster = []
        for i in self.mapping:
            if self.mapping[i] == m:
                cluster.append(i)
        return cluster

    def get_final_clusters(self, medoids):
        self.clusters = []

        for m in medoids:
            self.clusters.append(self.get_cluster(m))
        self.all_groupings.append(self.clusters)

    def inner_distance(self, c, cluster):
        #returns distance from element c to the rest of the elements in the cluster
        d = 0
        for i in cluster:
            d += self.distances[i, self.languages[c]]
        return d

    def initialize_clusters(self, medoids):
        # for every lannguage get the closest medoid
        # store information about clustering

        for l in self.languages:
            self.mapping[l] = -1

        #print(medoids)
        for i in self.languages:
            dist = 1
            ind = -1
            for m in medoids:
                j = self.languages[m]
                d = self.distances[i, j]
                # print('razdalja od', i, ' do ', j, 'je ', d)
                if d < dist:
                    dist = d
                    ind = m
            # print('min razdalja ' ,i, ' je do ', self.languages[ind])
            self.mapping[i] = ind

        #print(self.mapping)

    def determine_silhuette(self, a, b):
        for i in range(self.n):
            if a[i] == -1:
                self.silh[i] = 0
            else:
                self.silh[i] = (b[i] - a[i]) / max(a[i], b[i])

        avg_silhouette = np.mean(self.silh)
        self.average_silh.append(avg_silhouette)

        if avg_silhouette > self.max_silhouette:
            self.max_silhouette = avg_silhouette
            self.max_index = self.current_iteration
        if avg_silhouette < self.min_silhouette:
            self.min_silhouette = avg_silhouette
            self.min_index = self.current_iteration

        self.current_iteration += 1

        #print(self.silh)

    def silhuette(self, medoids):
        # a[i] is the distance of i to the rest of the elements in his own cluster
        a = [0] * self.n
        # b[i] is the distance of i to the elements of its closest neighbouring cluster j
        b = [0] * self.n

        for i in range(len(self.languages)):
            #inner distance
            point = self.languages[i]
            cluster = self.get_cluster(self.mapping[point])
            if len(cluster) > 1:
                a[i] = (self.inner_distance(i, cluster)) / (len(cluster)-1)
                #distance to other clusters, select the smallest
                dist = 100
                c = []
                for m in medoids:
                    other_cluster = self.get_cluster(m)
                    if m != self.mapping[point] and i not in other_cluster:
                        d = self.inner_distance(i, other_cluster)
                        #print('Distance from ', point, ' to ', other_cluster, 'is ', d)
                        if d < dist:
                            dist = d
                            c = other_cluster
                b[i] = dist / len(c)
            else:
                a[i] = -1
                b[i] = -1

        self.determine_silhuette(a, b)

    def draw_final_histograms(self):
        self.draw_histogram(self.average_silh)

    def draw_histogram(self, x):
        y_pos = np.arange(len(x))
        plt.bar(y_pos, x, align='center', alpha=0.5)
        plt.xticks(y_pos, rotation = 'vertical')
        plt.ylabel('Average silhouette values')
        plt.title('Histogram')
        plt.show()

    def k_medoids(self, k):

        medoids = self.random_no(k)
        #print(medoids)

        self.initialize_clusters(medoids)

        costs = [0] * k
        curr = [1] * k
        change = 1

        while 1:
            if change == 1:
                change = 0
            else:
                break
            for i in range(len(medoids)):
                cluster = self.get_cluster(medoids[i])
                costs[i] = self.inner_distance(medoids[i], cluster)
                #print('Cost for medoid ', medoids[i], ' is ', costs[i])
                curr[i] = costs[i]
                indc = -1
                for c in range(len(cluster)):
                    #calculate distance from c to the rest of elements in the cluster
                    dist = self.inner_distance(c, cluster)
                    if curr[i] > dist:
                        curr[i] = dist
                        indc = c
                #print('New low cost for ', medoids[i], ' is ', curr[i])
                if indc != -1:
                    #update current medoid
                    medoids[i] = indc
                    change = 1
            self.initialize_clusters(medoids)

        self.get_final_clusters(medoids)
        self.silhuette(medoids)
        #self.draw_histogram(self.silh)

    def recognize_language(self, file, dir):
        triples = read_language(file, dir)
        self.lan['text'] = triples
        dist = []
        sum = 0

        for i in range(len(self.languages)):
            v1, v2 = self.dict_intersection('text', self.languages[i])
            d = self.cos_distance(v1, v2)
            sum += (1-d)
            dist.append(d)

        indexed = list(enumerate(dist))
        top_3 = sorted(indexed, key=operator.itemgetter(1), reverse=True)
        lst = [k[0] for k in top_3][-3:]
        lst.reverse()

        print('Podano besedilo je najbolj podobno jezikom: ')
        for j in lst:
            p = ((1-dist[j]) / sum)*100
            print(self.languages[j], round(p,2), '%')


if __name__ == "__main__":
    files = ['slv.txt', 'src5.txt', 'mkj.txt', 'blg.txt', 'rus.txt', 'ruw.txt', 'spn.txt', 'por.txt', 'gln.txt',
             'frn.txt', 'itn.txt', 'ger.txt', 'eng.txt', 'dut.txt', 'fri.txt', 'nrn.txt', 'dns.txt', 'swd.txt',
             'grk.txt', 'aln.txt']
    languages = ['slv', 'src5', 'mkj', 'blg', 'rus', 'ruw', 'spn', 'por', 'gln', 'frn', 'itn', 'ger', 'eng', 'dut',
                 'fri', 'nrn', 'dns', 'swd', 'grk', 'aln']

    km = KMedoids(read_files(files, languages, 'ready\\'))
    km.diff_all_languages()

    iterations = 100
    for i in range(iterations):
        km.k_medoids(5)

    print('Najboljša povprečna silhueta:')
    print(km.all_groupings[km.max_index])
    print(km.max_silhouette)
    print('Najslabša povprečna silhueta:')
    print(km.all_groupings[km.min_index])
    print(km.min_silhouette)
    km.draw_final_histograms()
    print('Napovedovanje jezika iz ready/text.txt datoteko:')
    km.recognize_language('text.txt', 'ready\\')

#Na spletu najdite novičarske strani v prej izbranih dvajsetih jezikih in ponovite razvrščanje na novicah. Komentirajte rezultate.
    news = KMedoids(read_files(files, languages, 'news\\'))
    news.diff_all_languages()
    news.k_medoids(5)
    print('Razvrščanje na novicah')
    print(news.clusters)