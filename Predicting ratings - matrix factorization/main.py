import numpy as np
import random

class RecommendationSystem:
    def __init__(self):
        self.users_dict = dict()
        self.items_dict = dict()
        self.train = None
        self.test = None

    def read_file(self, file):
        data = [i.strip().split() for i in open(file).readlines()]
        data.pop(0)

        for l in data:
            u = l[0]
            i = l[1]
            if u not in self.users_dict:
                self.users_dict[u] = []
            self.users_dict[u].append(float(l[2]))
            if i not in self.items_dict:
                self.items_dict[i] = []
            self.items_dict[i].append(float(l[2]))

        #print(self.users_dict)
        #print(self.items_dict)
        return data

    def read_test(self, file):
        data = [i.strip().split() for i in open(file).readlines()]
        data.pop(0)
        return data

    def compute_averages(self):
        for u in self.users_dict:
            l = self.users_dict[u]
            self.users_dict[u] = sum(l) / len(l)

        for i in self.items_dict:
            l = self.items_dict[i]
            self.items_dict[i] = sum(l) / len(l)

    def predicting(self):

        izhod = open(r"averages.txt", "w+")

        for i in range(len(self.test)):
            u = self.test[i][0]
            i = self.test[i][1]
            scores = []
            if u in self.users_dict:
                scores.append(self.users_dict[u])
            if i in self.items_dict:
                scores.append(self.items_dict[i])
            pred = sum(scores) / len(scores)
            print(pred)
            izhod.write(str(pred) + '\n')

        izhod.close()




if __name__ == "__main__":
    R = RecommendationSystem()
    R.train = R.read_file("user_artists_training.dat")
    R.test = R.read_test("user_artists_test.dat")
    R.compute_averages()
    R.predicting()