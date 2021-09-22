import csv
import math
import numpy

def read_file(file_name):
    """
    Read and process data to be used for clustering.
    :param file_name: name of the file containing the data
    :return: dictionary with element names as keys and feature vectors as values
    """
    f = open("eurovision-finals-1975-2019.csv", "rt", encoding="utf8")

    #   Get a sorted list of all countries and all years
    countries = set()
    years = set()
    #   country i gives (on average) n votes to country j, in year k
    country_votes = dict([])
    #   how many times has the voting occurred
    voting_times = dict([])
    data = []
    for l in csv.reader(f):
        #clean data
        if l[2] == 'F.Y.R. Macedonia':
            l[2] = 'North Macedonia'

        if l[3] == 'F.Y.R. Macedonia':
            l[3] = 'North Macedonia'

        if l[4] == '10':
           l[4] = '9'
        if l[4] == '12':
            l[4] = '10'

        if l[2] != 'Yugoslavia' and l[2] != 'Serbia & Montenegro':
            if l[3] != 'Yugoslavia' and l[3] != 'Serbia & Montenegro':
                data.append([l[0], l[2], l[3], l[4]])

    data.remove(data[0])

    for l in data:
        countries.add(l[1])
        years.add(l[0])
        country_votes[l[0], l[1], l[2]] = -1
        voting_times[l[0], l[1], l[2]] = 0

    countries = list(countries)
    years = list(years)
    countries.sort()
    years.sort()

    for row in data:
        if country_votes[row[0], row[1], row[2]] == -1:
            country_votes[row[0], row[1], row[2]] = int(row[3])
        else:
            country_votes[row[0], row[1], row[2]] += int(row[3])
        voting_times[row[0], row[1], row[2]] += 1

    for i, j, k in country_votes:
        if country_votes[i, j, k] == -1:
            continue
        else:
            country_votes[i, j, k] = round(country_votes[i, j, k]/voting_times[i, j, k], 3)

    result = dict([])

    for country in countries:
        result[country] = numpy.full((len(countries) * len(years)), -1)

    for i, j, k in country_votes:
        result[j][countries.index(k) + (len(countries) * years.index(i))] = country_votes[i, j, k]

    return result

def read_file_jury_televoting(file_name, parameter):

    f = open("eurovision-finals-1975-2019.csv", "rt", encoding="utf8")

    data = []

    for l in csv.reader(f):
        if l[1] == parameter:
            if int(l[0]) >= 2016:
                if l[2] == 'F.Y.R. Macedonia':
                    l[2] = 'North Macedonia'

                if l[3] == 'F.Y.R. Macedonia':
                    l[3] = 'North Macedonia'

                if l[4] == '10':
                    l[4] = '9'
                if l[4] == '12':
                    l[4] = '10'

                data.append([l[0], l[2], l[3], l[4]])

    countries = set()
    country_votes = dict([])
    voting_times = dict([])

    for l in data:
        countries.add(l[1])
        country_votes[l[1], l[2]] = 0
        voting_times[l[1], l[2]] = 0

    countries = list(countries)
    countries.sort()

    for row in data:
        country_votes[row[1], row[2]] += int(row[3])
        voting_times[row[1], row[2]] += 1

    for i, j in country_votes:
        if country_votes[i, j] == 0:
            continue
        else:
            country_votes[i, j] = round(country_votes[i, j] / voting_times[i, j], 3)

    result = dict([])

    for country in countries:
        vector = numpy.zeros(len(countries))
        for i, j in country_votes:
            if country == i:
                vector[countries.index(j)] = country_votes[i, j]

        result[country] = vector
        # print("From country:", country, " to countries => ", vector)

    return result


def show_statistics(self, cluster, averages, all_countries):

        low_statistics = dict ([])
        high_statistics = dict ([])

        for country in averages:
            low_statistics[country] = 0
            high_statistics[country] = 0

        for country in cluster:
            print("Votes from ", country, ":")

            for i in range(len(all_countries)):
                if averages[country][i] >= 6:
                    print("High points: " + all_countries[i])
                    high_statistics[all_countries[i]] += 1
                if averages[country][i] <= 2:
                    print("Low points: " + all_countries[i])
                    low_statistics[all_countries[i]] += 1

        print(high_statistics)
        print(low_statistics)

class HierarchicalClustering:
    def __init__(self, data):
        """Initialize the clustering"""
        self.data = data
        # self.clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into clusterings of the type
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        self.clusters = [[name] for name in self.data.keys()]
        self.final_clusters = [[name] for name in self.data.keys()]


    def row_distance(self, r1, r2):
        """
        Distance between two rows.
        Implement either Euclidean or Manhattan distance.
        Example call: self.row_distance("Polona", "Rajko")
        """
        sum = 0
        n = 0
        max_distance = 10000000
        for i in range(len(self.data[r1])):
            if self.data[r1][i] != -1 and self.data[r2][i] != -1:
                sum += ((self.data[r1][i] - self.data[r2][i]) ** 2)
                n += 1

        if n == 0:
            return max_distance

        sum = (sum/n) * len(self.data[r1])

        return math.sqrt(sum)

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
                #sum += (self.row_distance(pi, pj) ** 2)
                sum += (self.row_distance(pi, pj))

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

        #at the beginning, every country represents a cluster
        no_clusters = len(self.clusters)

        while no_clusters > 1:

            #find the closest clusters and store information
            x, y, d = self.closest_clusters()

            #merge them as one, delete the separate ones
            self.clusters.remove(x)
            self.clusters.remove(y)
            self.clusters.append([x, y])

            #store clusters with distance below threshold
            if d <= 162:
                self.final_clusters.remove(x)
                self.final_clusters.remove(y)
                self.final_clusters.append([x, y])

            no_clusters -= 1

        #print('The final clusters are: ', self.final_clusters)
        self.clusters = self.clusters[0]

    def plot_tree(self):
        """
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """
        self.rek(self.clusters, 0)

if __name__ == "__main__":
    DATA_FILE = "eurovision-final.csv"
    hc = HierarchicalClustering(read_file(DATA_FILE))
    hc.run()
    hc.plot_tree()

    #Example how we get the two dendrograms for jury and televoting
    jury = HierarchicalClustering(read_file_jury_televoting(DATA_FILE, 'J'))
    jury.run()
    jury.plot_tree()
    televoting = HierarchicalClustering(read_file_jury_televoting(DATA_FILE, 'T'))
    televoting.run()
    televoting.plot_tree()