import lpputils
import linear
import gzip
import csv
import numpy as np
import pandas as pd

def read_data(file):
    reader = csv.reader(gzip.open(file, "rt"), delimiter="\t")
    header = next(reader)

    linije = dict([])
    vseLinije = []
    for r in reader:
        if r[3] not in linije:
            linije[r[3]] = []
            vseLinije.append(r[3])
        linije[r[3]].append([r[6], r[8]])

    return vseLinije, linije

def read_test(file):
    reader = csv.reader(gzip.open(file, "rt"), delimiter="\t")
    header = next(reader)

    rez = []
    bus = []
    for r in reader:
        rez.append([r[6], r[8]])
        bus.append(r[3])
    return rez, bus

def weather():

    dez_sneg = {}

    df = pd.read_csv(r'ljubljana_2012.csv')
    for i in range(df.shape[0]):
        d = df['dez'][i]
        s = df['sneg'][i]
        if d.isspace():
            d = '0'
        if s.isspace():
            s = '0'
        r = max(float(d), float(s))
        if r > 5:
            dez_sneg[df['dan'][i]] = 1
        else:
            dez_sneg[df['dan'][i]] = 0

    return dez_sneg

def prazniki():
    praznik = {}
    for i in range(1, 32):
        for j in range(1, 13):
            praznik[(i, j)] = 0

    praznik[(1, 1)] = 1
    praznik[(2, 1)] = 1
    praznik[(8, 2)] = 1
    praznik[(9, 4)] = 1
    praznik[(27, 4)] = 1
    praznik[(9, 4)] = 1
    praznik[(1, 5)] = 1
    praznik[(2, 5)] = 1
    praznik[(31, 5)] = 1
    praznik[(25, 6)] = 1
    praznik[(15, 8)] = 1
    praznik[(31, 10)] = 1
    praznik[(1, 11)] = 1
    praznik[(25, 12)] = 1
    praznik[(26, 12)] = 1
    return praznik


def generate(data):
    y = []
    odhod = []

    for line in data:
        if line[1] != '?':
            y.append(lpputils.tsdiff(line[1], line[0]))
        odhod.append(lpputils.parsedate(line[0]))

    X = np.zeros(shape=(len(odhod), 489))
    p = prazniki()
    vreme = weather()

    for i in range(len(odhod)):
        ind = 20 * odhod[i].hour + odhod[i].minute // 3
        X[i][ind] = 1
        X[i][odhod[i].weekday() + 480] = 1
        X[i][487] = p[(odhod[i].day, odhod[i].month)]
        datum = str(odhod[i].month) + '/' + str(odhod[i].day) + '/' + str(odhod[i].year)
        X[i][488] = vreme[datum]

    return X, np.array(y), odhod

if __name__ == "__main__":

    vseLinije, linije  = read_data("vseLinije/train.csv.gz")

    napovednik = dict([])
    lr = linear.LinearLearner(lambda_=1.)

    for l in vseLinije:
        X, y, odhod = generate(linije[l])
        napovednik[l] = lr(X, y)

    data, bus = read_test("vseLinije/test.csv.gz")
    Xtest, y, odhod = generate(data)

    izhod = open(r"rezultati.txt", "w+")
    for i in range(len(Xtest)):
        print(lpputils.tsadd(odhod[i], napovednik[bus[i]](Xtest[i])))
        izhod.write(lpputils.tsadd(odhod[i], napovednik[bus[i]](Xtest[i])) + '\n')

    izhod.close()