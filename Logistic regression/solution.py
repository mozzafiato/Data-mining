import numpy as np
import math
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import confusion_matrix


def load(name):
    """
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke)
    in vektor razredov.
    """
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y

def h(x, theta):
    """
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    # ... dopolnite (naloga 1)
    koeficient = np.dot(theta, x)
    e = 1 / (1 + math.exp(-koeficient))

    return e

def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    v = 0

    for i in range(len(X)):
        h0 = h(X[i], theta)
        if y[i] == 1:
            v += (y[i] * math.log(h0))
        else:
            v += ((1 - y[i]) * math.log(1 - h0))

    return (-1) / len(X) * v + np.dot(theta, theta) * lambda_


def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne 1D numpy array v velikosti vektorja theta.
    """
    rez = np.zeros(len(theta))

    for i in range(len(X)):
        h0 = h(X[i], theta)
        n = h0 - y[i]
        for j in range(len(X[0])):
            rez[j] += (n * X[i, j])

    return rez * (1/len(X)) + 2*theta*lambda_

def num_grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    # ... dopolnite (naloga 1, naloga 2)
    rez = np.zeros(len(theta))

    konstanta = 0.00001

    for i in range(len(theta)):
        t1 = theta
        t1[i] += konstanta/2
        c1 = cost(t1, X, y, lambda_)

        t1[i] -= konstanta
        c2 = cost(t1, X, y, lambda_)

        t1[i] += konstanta / 2

        rez[i] = (c1 - c2) / konstanta

    return rez


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetno razreda 1
        return [1-p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = np.hstack((np.ones((len(X),1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, k=5):
    """
    Primer klica:
        res = test_cv(LogRegLearner(lambda_=0.0), X, y)
    ... dopolnite (naloga 3)
    """

    n = len(X)
    rez = np.zeros(shape=(n, 2))

    shuffled = np.arange(n)
    np.random.shuffle(shuffled)
    splits = np.array_split(shuffled, k)

    indexes = np.arange(n)
    for i in range(k):
        v_indexes = splits[i]
        t_indexes = np.delete(indexes, v_indexes)

        t_X = X[t_indexes]
        t_y = y[t_indexes]

        learner_cv = learner(t_X, t_y)

        v_X = X[v_indexes]
        for j in range(len(v_indexes)):
            rez[v_indexes[j], :] = learner_cv(v_X[j])

    return rez

def CA(real, predictions):
    return 1-sum(abs(real-np.argmax(predictions, axis=1)))/len(real)


def AUC(real, predictions, lambda_=0.0):

    n = len(real)
    learner = LogRegLearner(lambda_)
    classifier = learner(predictions, real)

    # prepare tabela = [real, predictions p(y=1|x)]
    tabela = np.zeros(shape=(n, 2))

    for i in range(n):
        p = classifier(predictions[i])
        tabela[i][0] = real[i]
        tabela[i][1] = p[1]

    # sort data
    tabela = tabela[tabela[:, 1].argsort()[::-1]]

    # get coordinates
    dots = []
    threshold = 1
    step = 0.01
    for i in range(int(1/step)+1):
        pred = np.zeros(n)
        pred[tabela[:, 1] > threshold] = 1

        #print(pred)
        tn, fp, fn, tp = confusion_matrix(tabela[:, 0], pred).ravel()
        x = tp / (tp + fn)
        y = 1 - (tn / (tn + fp))
        dots.append([y, x])

        threshold -= step

    area = 0
    for i in range(len(dots)-1):
        # (x2 - x1) * (y2 - y1) / 2  + y1 * (x2 - x1)
        area += ((dots[i+1][0] - dots[i][0]) * (dots[i+1][1] - dots[i][1])) / 2
        area += dots[i][1] * (dots[i+1][0]-dots[i][0])

    return area


if __name__ == "__main__":

    X, y = load('reg.data')

    lambda_values = [5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.0, 0.1, 0.01, 0.001]

    for l in lambda_values:
        learner = LogRegLearner(lambda_=l)
        classifier = learner(X, y)

        res = test_cv(learner, X, y)
        cv_ocena = round(CA(y, res), 3)
        learning_ocena = round(CA(y, test_learning(learner, X, y)), 3)
        auc_ocena = round(AUC(y, X, l), 3)

        #print(l, " &", learning_ocena, " & ", cv_ocena, " & ", auc_ocena, " \\\ ")
        print("Lambda: ", l, " TR: ", learning_ocena, " CV: ", cv_ocena, " AUC: ", auc_ocena)