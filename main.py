import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from math import log
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import tree as tre
from kneed import KneeLocator
from collections import Counter
import joblib
from pprint import pprint



label_encoder = LabelEncoder()  # creating labelEncoder


def get_numerical_df(df):
    """
    :param df: mixed data frame
    :return: data frame containing numeric columns
    """
    int_str = 'int'
    float_str = 'float'
    numerical_columns = []
    for col in df:
        if df[col].dtype.name[:3:] == int_str or df[col].dtype.name[:5:] == float_str:
            numerical_columns.append(col)
    return pd.DataFrame(df[numerical_columns]), numerical_columns


def get_numerical_cols(df):
    """
    :param df: mixed data frame
    :return: numeric columns of data frame
    """
    a = df.describe()
    return df[a.columns]


def rootInit():
    """
    set the main screen
    :return: main screen
    """
    root = Tk()
    root.title("Data Mining")  # set title
    root.minsize(150, 100)  # set min size
    root.configure(bg='LightBlue')  # set background
    return root


def callbackTrain():
    """ load train csv file """
    csv_file_path = fd.askopenfilename(title="Select file", filetypes=(("CSV Files", "*.csv"),))
    if csv_file_path is not None:
        myLabel.configure(text='The file you selected is:\n' + csv_file_path + '\ndo you want to choose another one?',
                          font=('Arial Bold', 20))
        global train, fileTrain
        train = pd.read_csv(csv_file_path)
        fileTrain = True
        nextButton()


def callbackTest():
    """ load test csv file """
    csv_file_path = fd.askopenfilename(title="Select file", filetypes=(("CSV Files", "*.csv"),))
    if csv_file_path is not None:
        myLabel.configure(text='The file you selected is:\n' + csv_file_path + '\ndo you want to choose another one?',
                          font=('Arial Bold', 20))
        global test, fileTest
        test = pd.read_csv(csv_file_path)
        fileTest = True
        nextButton()


def nextButton():
    """ enable next button """
    if fileTest and fileTrain:
        next_button.configure(state=NORMAL)


def startMining():
    """ Displays and allows the user to choose which algorithm to perform """
    new_window = rootInit()
    file_window.destroy()
    notebook = ttk.Notebook(new_window)
    tab1 = Frame(notebook, bg='LightBlue')
    tab2 = Frame(notebook, bg='LightBlue')
    tab4 = Frame(notebook, bg='LightBlue')

    def setTabs():
        """ set the displays to choose which algorithm to perform """
        Label(tab1, text="Discretization choices: ", padx=25, font=("Impact", 20), bg='LightBlue').pack()
        Discretization = ["width-Equal", "width-Equal Python", "Equal-Frequency", "Equal-Frequency Python",
                          "by Entropy"]

        def getDiscretizationChoices():
            """ :return the user choice which algorithm to perform """
            if discVar.get() <= len(Discretization):
                return Discretization[discVar.get()]
            else:
                return Discretization[0]

        discVar = IntVar(tab1)
        discVar.set(0)
        for index in range(len(Discretization)):
            Radiobutton(tab1,
                        text=Discretization[index],
                        variable=discVar,
                        value=index,
                        padx=25,
                        font=("Impact", 20),
                        bg='LightBlue',
                        command=getDiscretizationChoices).pack(anchor=W)
        Label(tab1, text="amount of bins:", padx=25, font=("Impact", 20), bg='LightBlue').pack()
        scale = Scale(tab1, from_=2, to=6, bg='LightBlue', orient=HORIZONTAL, width=30, length=280)
        scale.pack(anchor=S)

        Label(tab2, text="Normalization choices: ", padx=10, font=("Impact", 15), bg='LightBlue').pack()
        Normalization = ["MinMaxScaler", "MinMaxScaler Python", "zScore", "zScore Python", "Decimal scale",
                         "without normalization"]

        def getNormalizationChoices():
            """ :return the user choice which algorithm to perform """
            if normVar.get() <= len(Normalization):
                return Normalization[normVar.get()]
            else:
                return Normalization[0]

        normVar = IntVar(tab2)
        normVar.set(0)
        for index in range(len(Normalization)):
            Radiobutton(tab2,
                        text=Normalization[index],
                        variable=normVar,
                        value=index,
                        padx=25,
                        font=("Impact", 20),
                        bg='LightBlue',
                        command=getNormalizationChoices
                        ).pack(anchor=W)

        Label(tab4, text="null values choices: ", padx=25, font=("Impact", 20), bg='LightBlue').pack()
        isnull = ["dropna", "commonOrMeanValue"]

        def getIsnullChoices():
            """ :return the user choice which algorithm to perform """
            if isnullVar.get() <= len(isnull):
                return isnull[isnullVar.get()]
            else:
                return isnull[0]

        isnullVar = IntVar(tab4)
        isnullVar.set(0)

        for index in range(len(isnull)):
            Radiobutton(tab4,
                        text=isnull[index],
                        variable=isnullVar,
                        value=index,
                        padx=25,
                        font=("Impact", 20),
                        bg='LightBlue',
                        command=getIsnullChoices).pack(anchor=W)

        def level2Mining():
            """ Displays and allows the user to choose which algorithm to perform """

            new_window2 = rootInit()
            Label(new_window2, text="Choose your required algorithm: ", padx=25, font=("Purisa", 20),
                  bg='LightBlue').pack()
            alg = ["naive bayes", "naive bayes Python", "Decision tree", "Decision tree Python", "Kmeans", "knn"]

            def getAlgChoices():
                """ :return the user choice which algorithm to perform """
                if algVar.get() <= len(alg):
                    doPreAlgorithm([getDiscretizationChoices(), scale.get()], getNormalizationChoices(),
                                   getIsnullChoices(),
                                   alg[algVar.get()])
                else:
                    return 0

            algVar = IntVar(new_window2)
            algVar.set(0)
            for ind in range(len(alg)):
                Radiobutton(new_window2,
                            text=alg[ind],
                            variable=algVar,
                            value=ind,
                            padx=25,
                            font=("Purisa", 15),
                            bg='LightBlue'
                            ).pack(anchor=W)

            next_button3 = Button(new_window2, text='Next',
                                  command=getAlgChoices,
                                  font=("comic sans", 20),
                                  state=NORMAL
                                  )
            next_button3.pack(side=RIGHT)
            cls_button = Button(new_window2, text='Close',
                                font=("comic sans", 20),
                                command=new_window2.destroy).pack(side=LEFT)
            new_window2.mainloop()

        next_button2 = Button(tab4, text='Next',
                              command=level2Mining,
                              font=("comic sans", 20),
                              state=NORMAL
                              )
        next_button2.pack(side=RIGHT)

    notebook.add(tab1, text="Discretization")
    notebook.add(tab2, text="Normalization")
    notebook.add(tab4, text="Null values")
    notebook.pack(expand=True, fill='both')
    setTabs()
    new_window.mainloop()


def doPreAlgorithm(disc, norm, null, alg):
    """ perform by the user choice pre-algorithm """
    global train, test, dfTrain, dfTest
    dfTrain = train.copy()
    dfTest = test.copy()

    def combined_same_value_na(df):
        """ Delete columns with a classification column with no value """
        cols = list(df.columns)
        for col in df:
            if len(list(df[col].value_counts())) >= 1:
                if df[col].isna().sum() + list(df[col].value_counts())[0] == len(df[col]):
                    cols.remove(col)
            else:
                if df[col].isna().sum() == len(df[col]):
                    cols.remove(col)
        return df[cols]

    def doNullValues(null):
        """ perform by the user choice complete missing values """
        global dfTrain, dfTest

        def find_common(col):
            """ find the most common value in the col """
            dic = dict(col.value_counts())
            lst = list(dic.items())
            for i in range(len(lst)):
                if lst[i][1] == max(col.value_counts()):
                    return lst[i][0]

        def get_values(df):
            """ return the most common or mean value in each col """
            value_dict = dict()
            cols = df.columns
            for col in df:
                value_dict[str(col)] = 0
            for i in range(len(df.columns)):
                if cols[i] in df.describe().columns:
                    value_dict[cols[i]] = df[cols[i]].mean()
                else:
                    value_dict[cols[i]] = find_common(df[cols[i]])
            return value_dict

        def my_fill_na(df):
            """ complete missing values by most common or mean value in each col """
            values = get_values(df)
            df.fillna(value=values, inplace=True)
            return df

        if null == "dropna":
            dfTrain.dropna(inplace=True)
            dfTest.dropna(inplace=True)
        elif null == "commonOrMeanValue":
            dfTrain = my_fill_na(dfTrain)
            dfTest = my_fill_na(dfTest)

    def doDiscretization(disc):
        """
            perform by the user choice discretization
            :param disc is array length 2
                   the first element is string and the second is the amount of bins
        """

        global dfTrain, dfTest

        def encoded_df(df):
            """ :return data frame after label encoding columns """
            for col in df:
                df[col] = label_encoder.fit_transform(df[col].astype(str))
            return df

        def PythonEqualFrequency(df, b):
            class_col = df.columns[::-1][0]
            X_df = get_numerical_cols(df)
            for col in X_df:
                equal_freq_col = pd.qcut(df[col], b, duplicates='drop')
                df[col] = equal_freq_col
            return encoded_df(df)

        def PythonWidthEqual(df, b):
            class_col = df.columns[::-1][0]
            X_df = get_numerical_cols(df)
            for col in X_df:
                equal_freq_col = pd.cut(df[col], b)
                df[col] = equal_freq_col
            return encoded_df(df)

        def myEqualFrequencyDf(df, b):
            def myEqualFrequency(arr1, b):
                temp = [list(arr1)]
                temp[0].sort()
                lst = []
                a = len(temp[0])
                n = int(a / b)
                for i in range(0, b):
                    arr = []
                    for j in range(i * n, (i + 1) * n):
                        if j >= a:
                            break
                        arr = arr + [temp[0][j]]
                    lst.append(arr)
                return lst

            def update_array_to_discret(arr, labels, maximum_arr):
                temp = [list(arr)]
                size = len(labels)
                for i in range(len(temp)):
                    for j in range(len(temp[i])):
                        for k in range(size):
                            if temp[i][j] <= maximum_arr[k]:
                                temp[i][j] = labels[k]
                                break
                return temp

            def get_labels(lst):
                minimum_lst = list()
                maximum_lst = list()
                labels = list()
                for i in range(len(lst)):
                    minimum_lst.append(min(lst[i]))
                    maximum_lst.append(max(lst[i]))
                    labels.append(str(min(lst[i])) + ", " + str(max(lst[i])))
                return labels, maximum_lst

            X_df = get_numerical_cols(df)
            for col in X_df:
                temp = myEqualFrequency(df[str(col)], b)
                labels, maximum = get_labels(temp)
                categorial_col = update_array_to_discret(df[str(col)], labels, maximum)
                df[str(col)] = categorial_col[0]
            return encoded_df(df)

        def myWidthEqualDf(df, b):
            def get_labels_for_width(arr, k):
                """ :return 2 arrays contain labels and max values in each array """
                lst = list(arr)
                w = (max(lst) - min(lst)) / k
                temp = []
                maximum_list = []
                min_val = min(lst)
                for i in range(k):
                    maximum_list.append(min_val + w)
                    temp.append(str(min_val) + ", " + str(min_val + w))
                    min_val = min_val + w
                return temp, maximum_list

            def my_equal_width(arr, labels, maximum):
                lst = list(arr)
                for i in range(len(lst)):
                    for k in range(len(labels)):
                        if lst[i] <= maximum[k]:
                            lst[i] = labels[k]
                            break
                return lst

            X_df = get_numerical_cols(df)
            for col in X_df:
                labels, maximum = get_labels_for_width(X_df[col], b)
                cat_col = my_equal_width(df[col], labels, maximum)
                df[col] = cat_col
            return encoded_df(df)

        def myEntropyDiscrete(df):
            """ :return data frame after entropy based discretization """

            def combined_for_entropy(df):
                def entropy_discret(df, col):
                    def calc_pd_entropy(column):
                        """ Calculate entropy given a pandas series, list, or numpy array. """
                        # Compute the counts of each unique value in the column
                        counts = np.bincount(column)
                        # Divide by the total column length to get a probability
                        probabilities = counts / len(column)

                        # Initialize the entropy to 0
                        entropy = 0
                        # Loop through the probabilities, and add each one to the total entropy
                        for prob in probabilities:
                            if prob > 0:
                                # use log from math and set base to 2
                                entropy += prob * log(prob, 2)

                        return -entropy

                    class_col = df.columns[::-1][0]
                    main_entropy = calc_pd_entropy(df[class_col])
                    items_list = []
                    for i in range(len(df[str(col)]) - 1):
                        items_list.append((df[str(col)][i] + df[str(col)][i + 1]) / 2)
                    weighted_list = []
                    for i in range(len(items_list)):
                        first_entropy = calc_pd_entropy(df.loc[df[str(col)] < items_list[i]][class_col])
                        second_entropy = calc_pd_entropy(df.loc[df[str(col)] >= items_list[i]][class_col])
                        info = (len(df.loc[df[str(col)] < items_list[i]]) / len(df)) * first_entropy + (
                                len(df.loc[df[str(col)] >= items_list[i]]) / len(df)) * second_entropy
                        weighted_list.append(main_entropy - info)
                    index = weighted_list.index(max(weighted_list))
                    return items_list[index]

                work_df = get_numerical_cols(df)
                entropy_split_values = []
                for col in work_df:
                    entropy_split_values.append(entropy_discret(work_df, col))
                return entropy_split_values

            def entropy_based_discretization_by_me(df, lst):
                """ :return Encoding data frame after entropy based discretization """

                def turn_arr_to_label(arr, label):
                    """ :return array by labels """
                    temp = []
                    for i in range(len(arr)):
                        if arr[i] >= label:
                            temp.append('>=' + str(label))
                        else:
                            temp.append('<' + str(label))
                    return temp

                def get_entropy_labels(lst):
                    """ :return array contain labels by entropy """
                    labels = []
                    for i in range(len(lst)):
                        labels.append('>=' + str(lst[i]))
                        labels.append('<' + str(lst[i]))
                    return labels

                i = 0
                labels = get_entropy_labels(lst)
                temp = []
                numer_df = get_numerical_cols(df)
                for col in numer_df:
                    col_list = list(df[col])
                    temp = turn_arr_to_label(col_list, lst[i])
                    df[col] = temp
                    i = i + 1
                return encoded_df(df)

            X_df = pd.DataFrame(df[df.columns[:-1]])
            lst = combined_for_entropy(get_numerical_cols(X_df))
            return entropy_based_discretization_by_me(df, lst)

        if disc[0] == "width-Equal":
            dfTrain = myWidthEqualDf(dfTrain, disc[1])
            dfTest = myWidthEqualDf(dfTest, disc[1])
        elif disc[0] == "width-Equal Python":
            dfTrain = PythonWidthEqual(dfTrain, disc[1])
            dfTest = PythonWidthEqual(dfTest, disc[1])
        elif disc[0] == "Equal-Frequency":
            dfTrain = myEqualFrequencyDf(dfTrain, disc[1])
            dfTest = myEqualFrequencyDf(dfTest, disc[1])
        elif disc[0] == "Equal-Frequency Python":
            dfTrain = PythonEqualFrequency(dfTrain, disc[1])
            dfTest = PythonEqualFrequency(dfTest, disc[1])
        elif disc[0] == "by Entropy":
            dfTrain = myEntropyDiscrete(dfTrain)
            dfTest = myEntropyDiscrete(dfTest)

    def doNormalization(norm):
        """ perform by the user choice normalization """
        global dfTrain, dfTest

        def myMinMaxScaler(df):
            result = df.copy()
            for col in result.columns:
                result[col] = (result[col] - result[col].min()) / (result[col].max() - result[col].min())
            return result

        def PythonMinMaxScaler(df):
            scalar = MinMaxScaler()
            scalar.fit(df)
            result = pd.DataFrame(scalar.transform(df))
            d = {}
            print(df.columns.values)
            print(result.columns.values)

            for i in range(len(df.columns.values)):
                d[result.columns.values[i]] = str(df.columns.values[i])
            print(d)
            result.rename(columns=d, inplace=True)
            return result

        def my_zScore(df):
            result = df.copy()  # copy the dataframe
            for col in result.columns:  # apply the z-score method
                result[col] = (result[col] - result[col].mean()) / result[col].std()
            return result

        def Python_zScore(df):
            result = df.copy()
            return result.apply(stats.zscore)

        def my_decimalScale(df):
            result = df.copy()
            for x in df:
                p = result[x].max()
                q = len(str(abs(p)))
                result[x] = result[x] / 10 ** q
            return result

        if norm == "MinMaxScaler":
            dfTrain = myMinMaxScaler(get_numerical_df(dfTrain)[0])
            dfTest = myMinMaxScaler(get_numerical_df(dfTest)[0])
        elif norm == "MinMaxScaler Python":
            dfTrain = (PythonMinMaxScaler(get_numerical_df(dfTrain)[0]))
            dfTest = (PythonMinMaxScaler(get_numerical_df(dfTest)[0]))
        elif norm == "zScore":
            dfTrain = (my_zScore(get_numerical_df(dfTrain)[0]))
            dfTest = (my_zScore(get_numerical_df(dfTest)[0]))
        elif norm == "zScore Python":
            dfTrain = (Python_zScore(get_numerical_df(dfTrain)[0]))
            dfTest = (Python_zScore(get_numerical_df(dfTest)[0]))
        elif norm == "Decimal scale":
            dfTrain = (my_decimalScale(get_numerical_df(dfTrain)[0]))
            dfTest = (my_decimalScale(get_numerical_df(dfTest)[0]))

    dfTrain = combined_same_value_na(dfTrain)
    dfTest = combined_same_value_na(dfTest)
    doNullValues(null)
    doDiscretization(disc)
    doNormalization(norm)

    dfTrain.to_csv('train_clean.csv')
    dfTest.to_csv('test_clean.csv')

    doAlgorithm(alg)


def doAlgorithm(alg):
    """ perform by the user choice an algorithm """

    global train, test, dfTrain, dfTest

    def my_naive_bayes():
        class NB:
            def __init__(self, target, dataframe):
                self.df = dataframe
                # Target/Category Column
                self.c_n = target
                # Column Names
                self.cols = list(self.df.columns)
                self.cols.remove(self.c_n)

                # Determine Continuous or Discrete for each Columns
                self.rv = {}
                self.determine_rv_for_all()

                # Likelihoods of Discrete Random Variables
                self.store = {}
                self.discrete_likelihood_for_all()

            def discrete_likelihood_cal(self, x, y, z):
                """
                x -> Column Name (String)
                y -> Column Value (String)
                z -> Class value (String)
                c_n -> Class Name (Target) # Not an Argument here #

                Returns -> P(x = y | c_n = z)
                """
                df = self.df

                if x not in self.cols:
                    raise KeyError("Feature(column) not present in the Training Dataset")

                res = (1 + len(df[(df[x] == y) & (df[self.c_n] == z)])) / (
                        len(df[df[self.c_n] == z]) + len(df[x].unique()))

                """if res == 0.0:
                    return 1/(len(df[df[self.c_n] == z]) + len(df[x].unique()))"""

                return res

            def discrete_likelihood_for_all(self):
                df = self.df

                discrete_cols = [x for x in self.cols if self.rv[x] == 'discrete']

                dict1 = {}
                for x in discrete_cols:
                    dict2 = {}
                    for y in df[x].unique():
                        dict3 = {}
                        for z in df[self.c_n].unique():
                            # print('P({}="{}"|{}="{}") = {}'.format(x,y,self.c_n,z,self.discrete_likelihood_cal(x, y, z)))
                            dict3[z] = self.discrete_likelihood_cal(x, y, z)
                        dict2[y] = dict3
                    dict1[x] = dict2

                self.store = dict1

            def determine_rv(self, x):
                """
                x -> Column Name
                """
                df = self.df

                val = list(df[x])[0]

                if type(val) == str or (type(val) == int and len(df[x].unique()) < len(df[x])):
                    return 'discrete'
                return 'continuous'

            def determine_rv_for_all(self):
                """
                self.rv = {}
                """

                self.rv = {x: self.determine_rv(x) for x in self.cols}

            def normal_pdf(self, sample, x=None):
                mu = np.mean(sample)
                sigma = np.std(sample)
                if x == None:
                    x = sample

                expr = np.exp((-1 / 2) * (((x - mu) / sigma) ** 2)) / (np.sqrt(2 * np.pi * sigma))
                return expr

            def continuous_likelihood_cal(self, column_name, column_val, class_val):
                df = self.df

                sample = df[df[self.c_n] == class_val][column_name]

                return self.normal_pdf(sample, column_val)

            def likelihood_expr(self, class_val, expr):
                val = 1

                for k, v in expr:

                    if k not in self.cols:
                        raise KeyError("Feature(column) not present in the Training Dataset")

                    if self.rv[k] == 'discrete':
                        try:
                            store_val = self.store[k][v][class_val]
                        except:
                            store_val = self.discrete_likelihood_cal(k, v, class_val)
                    else:
                        store_val = self.continuous_likelihood_cal(k, v, class_val)

                    val *= store_val

                return val

            def prior(self, class_val):
                df = self.df
                return len(df[df[self.c_n] == class_val]) / df.shape[0]

            def predict(self, X):
                df = self.df

                if type(X) == pd.core.series.Series:
                    values_list = [list(X.items())]

                elif type(X) == pd.core.frame.DataFrame:
                    values_list = [list(y.items()) for x, y in X.iterrows()]

                else:
                    raise TypeError('{} is not supported type'.format(type(X)))

                predictions_list = []
                for values in values_list:
                    likelihood_priors = {}
                    for class_val in df[self.c_n].unique():
                        likelihood_priors[class_val] = self.prior(class_val) * self.likelihood_expr(class_val, values)
                    # print(likelihood_priors)

                    normalizing_prob = np.sum([x for x in likelihood_priors.values()])
                    probabilities = [(y / normalizing_prob, x) for x, y in likelihood_priors.items()]
                    # print(probabilities)

                    if len(probabilities) == 2:
                        # For 2 Class Predictions
                        max_prob = max(probabilities)[1]
                        predictions_list.append(max_prob)

                    else:
                        # For Mulit Class Predictions
                        exp_1 = [np.exp(x) for x, y in probabilities]
                        exp_2 = np.sum(exp_1)
                        softmax = exp_1 / exp_2
                        # print(softmax)
                        class_names = [y for x, y in probabilities]
                        softmax_values = [(x, y) for x, y in zip(softmax, class_names)]
                        # print(softmax_values)
                        max_prob = max(softmax_values)[1]
                        predictions_list.append(max_prob)

                return predictions_list

            def accuracy_score(self, X, Y):
                assert len(X) == len(Y), 'Given values are not equal in size'

                total_matching_values = [x == y for x, y in zip(X, Y)]
                return (np.sum(total_matching_values) / len(total_matching_values)) * 100

            def calculate_confusion_matrix(self, X, Y):
                df = self.df

                unique_class_values = df[self.c_n].unique()
                decimal_class_values = list(range(len(unique_class_values)))
                numerical = {x: y for x, y in zip(unique_class_values, decimal_class_values)}

                x = [numerical[x] for x in X]
                y = [numerical[y] for y in Y]

                n = len(decimal_class_values)
                confusion_matrix = np.zeros((n, n))

                for i, j in zip(x, y):
                    if i == j:
                        confusion_matrix[i][i] += 1
                    elif i != j:
                        confusion_matrix[i][j] += 1

                return confusion_matrix

            def precision_score(self, X, Y):
                """
                Implemented Only for Binary Classes

                X -> y_true
                Y -> y_pred
                """
                assert len(X) == len(Y), 'Given values are not equal in size'

                confusion_matrix = self.calculate_confusion_matrix(X, Y)
                tp = confusion_matrix[0][0]
                fp = confusion_matrix[1][0]

                return tp / (tp + fp)

            def recall_score(self, X, Y):
                """
                Implemented Only for Binary Classes

                X -> y_true
                Y -> y_pred
                """
                assert len(X) == len(Y), 'Given values are not equal in size'

                confusion_matrix = self.calculate_confusion_matrix(X, Y)
                tp = confusion_matrix[0][0]
                fn = confusion_matrix[0][1]

                return tp / (tp + fn)

        X_train = dfTrain[dfTrain.columns[:-1]]
        y_train = dfTrain[dfTrain.columns[-1]]
        X_test = dfTest[dfTest.columns[:-1]]
        y_test = dfTest[dfTest.columns[-1]]

        genx = NB(target=dfTrain.columns[-1], dataframe=dfTrain)
        y_pred = genx.predict(X_test)

        naive_bayes_joblib_model = GaussianNB()
        naive_bayes_joblib_model.fit(X_train, y_train)
        joblib.dump(naive_bayes_joblib_model, 'NaivebayesJoblib')
        NaivebayesJoblib = joblib.load('NaivebayesJoblib')
        jblb_score = NaivebayesJoblib.score(X_test, y_test)

        print('Accuracy Score -> {} %'.format(round(genx.accuracy_score(y_test, y_pred), 3)))
        print('Precison Score -> {}'.format(round(genx.precision_score(y_test, y_pred), 3)))
        print('Recall Score -> {}'.format(round(genx.recall_score(y_test, y_pred), 3)))
        print('by joblib Score -> {} %'.format(round(jblb_score * 100, 3)))

        clf = SVC(random_state=0)
        clf.fit(X_train, y_train.values.ravel())
        plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.cool)
        plt.show()
        report = classification_report(y_test, y_pred)
        print("Classification report:\n", report)

    def Python_naive_bayes():
        X_train = dfTrain[dfTrain.columns[:-1]]
        y_train = dfTrain[dfTrain.columns[-1]]
        X_test = dfTest[dfTest.columns[:-1]]
        y_test = dfTest[dfTest.columns[-1]]
        # Create a Gaussian Classifier
        gnb = GaussianNB()

        # Train the model using the training sets
        gnb.fit(X_train, y_train.values.ravel())

        # Predict the response for test dataset
        y_pred = gnb.predict(X_test)

        naive_bayes_joblib_model = GaussianNB()
        naive_bayes_joblib_model.fit(X_train, y_train)
        joblib.dump(naive_bayes_joblib_model, 'NaivebayesJoblib')
        NaivebayesJoblib = joblib.load('NaivebayesJoblib')
        jblb_score = NaivebayesJoblib.score(X_test, y_test)

        print("Gaussian Naive Bayes model accuracy(in %):", accuracy_score(y_test, y_pred) * 100)
        print('by joblib Score -> {} %'.format(round(jblb_score * 100, 3)))
        clf = SVC(random_state=0)
        clf.fit(X_train, y_train.values.ravel())
        plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.cool)
        plt.show()

        report = classification_report(y_test, y_pred)
        print("Classification report:\n", report)

    def myDecision_tree():
        def entropy(y):
            hist = np.bincount(y)
            ps = hist / len(y)
            return -np.sum([p * np.log2(p) for p in ps if p > 0])

        def frame_to_list(df):
            """ :return array of columns """
            temp = []
            for col in df:
                temp.append(df[col].to_list())
                b = np.array(temp)
            temp2 = []
            for i in range(len(temp[0])):
                temp2.append(b[:, i])
            return np.array(temp2)

        def accuracy(y_true, y_pred):
            accuracy = np.sum(y_true == y_pred) / len(y_true)
            return accuracy

        class Node:
            def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
                self.feature = feature
                self.threshold = threshold
                self.left = left
                self.right = right
                self.value = value

            def is_leaf_node(self):
                return self.value is not None

        class DecisionTree:
            def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
                self.min_samples_split = min_samples_split
                self.max_depth = max_depth
                self.n_feats = n_feats
                self.root = None

            def fit(self, X, y):
                self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
                self.root = self._grow_tree(X, y)

            def predict(self, X):
                return np.array([self._traverse_tree(x, self.root) for x in X])

            def _grow_tree(self, X, y, depth=0):
                n_samples, n_features = X.shape
                n_labels = len(np.unique(y))

                # stopping criteria
                if (
                        depth >= self.max_depth
                        or n_labels == 1
                        or n_samples < self.min_samples_split
                ):
                    leaf_value = self._most_common_label(y)
                    return Node(value=leaf_value)

                feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

                # greedily select the best split according to information gain
                best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

                # grow the children that result from the split
                left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
                left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
                right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
                return Node(best_feat, best_thresh, left, right)

            def _best_criteria(self, X, y, feat_idxs):
                best_gain = -1
                split_idx, split_thresh = None, None
                for feat_idx in feat_idxs:
                    X_column = X[:, feat_idx]
                    thresholds = np.unique(X_column)
                    for threshold in thresholds:
                        gain = self._information_gain(y, X_column, threshold)

                        if gain > best_gain:
                            best_gain = gain
                            split_idx = feat_idx
                            split_thresh = threshold

                return split_idx, split_thresh

            def _information_gain(self, y, X_column, split_thresh):
                # parent loss
                parent_entropy = entropy(y)

                # generate split
                left_idxs, right_idxs = self._split(X_column, split_thresh)

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    return 0

                # compute the weighted avg. of the loss for the children
                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
                child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

                # information gain is difference in loss before vs. after split
                ig = parent_entropy - child_entropy
                return ig

            def _split(self, X_column, split_thresh):
                left_idxs = np.argwhere(X_column <= split_thresh).flatten()
                right_idxs = np.argwhere(X_column > split_thresh).flatten()
                return left_idxs, right_idxs

            def _traverse_tree(self, x, node):
                if node.is_leaf_node():
                    return node.value

                if x[node.feature] <= node.threshold:
                    return self._traverse_tree(x, node.left)
                return self._traverse_tree(x, node.right)

            def _most_common_label(self, y):
                counter = Counter(y)
                most_common = counter.most_common(1)[0][0]
                return most_common

        def InfoGain(data, split_attribute_name, target_name="class"):
            total_entropy = entropy(data[target_name])
            vals, counts = np.unique(data[split_attribute_name], return_counts=True)
            Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(
                data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
            Information_Gain = total_entropy - Weighted_Entropy
            return Information_Gain

        def ID3(data, originaldata, features, parent_node_class=None):
            cols = originaldata.columns
            target_attribute_name = cols[::-1][0]
            if len(np.unique(data[target_attribute_name])) <= 1:
                return np.unique(data[target_attribute_name])[0]
            elif len(data) == 0:
                return np.unique(originaldata[target_attribute_name])[
                    np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
            elif len(features) == 0:
                return parent_node_class
            else:
                parent_node_class = np.unique(data[target_attribute_name])[
                    np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
                item_values = [InfoGain(data, feature, target_attribute_name) for feature in
                               features]  # Return the information gain values for the features in the dataset
                best_feature_index = np.argmax(item_values)
                best_feature = features[best_feature_index]
                tree = {best_feature: {}}
                features = [i for i in features if i != best_feature]

                for value in np.unique(data[best_feature]):
                    value = value
                    sub_data = data.where(data[best_feature] == value).dropna()
                    subtree = ID3(sub_data, originaldata, features, parent_node_class)
                    tree[best_feature][value] = subtree
                return (tree)

        df_test = test.copy()
        y_test = pd.DataFrame(df_test['class'])
        numer_test_df = get_numerical_cols(df_test)
        numer_with_y = pd.concat([numer_test_df, y_test], axis=1)

        numer_with_y['class'] = label_encoder.fit_transform(numer_with_y['class'])
        X_numer = numer_with_y.drop(['class'], axis=1)
        y_numer = numer_with_y['class']
        X_numer_list = frame_to_list(X_numer)
        y_numer_list = np.array(y_numer)
        X_train, X_test, y_train, y_test = train_test_split(X_numer_list, y_numer_list, test_size=0.2,
                                                            random_state=1234)
        clf = DecisionTree(max_depth=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy(y_test, y_pred)

        desicion_tree_joblib_model = tre.DecisionTreeClassifier(criterion='entropy')
        desicion_tree_joblib_model.fit(X_train, y_train)
        joblib.dump(desicion_tree_joblib_model, 'DesicionTreeJoblib')
        DesicionTreeJoblib = joblib.load('DesicionTreeJoblib')
        jblb_score = DesicionTreeJoblib.score(X_test, y_test)

        print("Accuracy:", acc * 100, "%")
        print('by joblib Score -> {} %'.format(round(jblb_score * 100, 3)))

        the_tree = ID3(numer_with_y, numer_with_y, numer_with_y.columns[:-1])
        pprint(the_tree)

        clf = SVC(random_state=0)
        clf.fit(X_train, y_train)
        plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Reds)
        plt.show()
        report = classification_report(y_test, y_pred)
        print("Classification report:\n", report)

    def PythonDecision_tree():
        def InfoGain(data, split_attribute_name, target_name="class"):
            def entropy(y):
                hist = np.bincount(y)
                ps = hist / len(y)
                return -np.sum([p * np.log2(p) for p in ps if p > 0])

            total_entropy = entropy(data[target_name])
            vals, counts = np.unique(data[split_attribute_name], return_counts=True)
            Weighted_Entropy = np.sum(
                [(counts[i] / np.sum(counts)) * entropy(
                    data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
                 for i in range(len(vals))])
            Information_Gain = total_entropy - Weighted_Entropy
            return Information_Gain

        def ID3(data, originaldata, features, parent_node_class=None):
            cols = originaldata.columns
            target_attribute_name = cols[::-1][0]
            if len(np.unique(data[target_attribute_name])) <= 1:
                return np.unique(data[target_attribute_name])[0]
            elif len(data) == 0:
                return np.unique(originaldata[target_attribute_name])[
                    np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
            elif len(features) == 0:
                return parent_node_class
            else:
                parent_node_class = np.unique(data[target_attribute_name])[
                    np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
                item_values = [InfoGain(data, feature, target_attribute_name) for feature in
                               features]  # Return the information gain values for the features in the dataset
                best_feature_index = np.argmax(item_values)
                best_feature = features[best_feature_index]
                tree = {best_feature: {}}
                features = [i for i in features if i != best_feature]

                for value in np.unique(data[best_feature]):
                    value = value
                    sub_data = data.where(data[best_feature] == value).dropna()
                    subtree = ID3(sub_data, originaldata, features, parent_node_class)
                    tree[best_feature][value] = subtree
                return (tree)

        def frame_to_list(df):
            """ :return array of columns """
            temp = []
            for col in df:
                temp.append(df[col].to_list())
                b = np.array(temp)
            temp2 = []
            for i in range(len(temp[0])):
                temp2.append(b[:, i])
            return np.array(temp2)

        df_train = train.copy()
        df_test = test.copy()

        df_train.dropna(inplace=True)
        df_test.dropna(inplace=True)

        X_train = frame_to_list(get_numerical_cols(df_train))
        y_train = df_train['class']

        tree = tre.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)
        tre.plot_tree(tree, fontsize=5)

        size = len(dfTrain.columns)

        X_train = frame_to_list(dfTrain)[:, range(size - 1)]
        y_train = frame_to_list(dfTrain)[:, -1]
        X_test = frame_to_list(dfTest)[:, range(size - 1)]
        y_test = frame_to_list(dfTest)[:, -1]

        tree = tre.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)
        prediction = tree.predict(X_test)

        desicion_tree_joblib_model = tre.DecisionTreeClassifier(criterion='entropy')
        desicion_tree_joblib_model.fit(X_train, y_train)
        joblib.dump(desicion_tree_joblib_model, 'DesicionTreeJoblib')
        DesicionTreeJoblib = joblib.load('DesicionTreeJoblib')
        jblb_score = DesicionTreeJoblib.score(X_test, y_test)

        print("The prediction accuracy is: ", tree.score(X_test, y_test) * 100, "%")
        print('by joblib Score -> {} %'.format(round(jblb_score * 100, 3)))

        clf = SVC(random_state=0)
        clf.fit(X_train, y_train)
        plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Reds)
        plt.show()

        report = classification_report(y_test, prediction)
        print("Classification report:\n", report)

    def Kmeans():
        def get_sse(df):
            sse = []
            for k in sse_range:
                km = KMeans(n_clusters=k)
                km.fit(df)
                sse.append(km.inertia_)
            return sse

        def Kmeans_by_me(original_df):
            colmap = {0: 'r', 1: 'g', 2: 'b', 3: 'y'}

            def split_data(lst):
                lst = np.array(lst)
                X_df = lst[:, range(len(lst[0]) - 1)]
                y_df = lst[:, -1]
                return X_df, y_df

            def frame_to_list(df):
                temp = []
                for col in df:
                    temp.append(df[col].to_list())
                    b = np.array(temp)
                temp2 = []
                for i in range(len(temp[0])):
                    temp2.append(b[:, i])
                return np.array(temp2)

            def update(df, centroids):

                for p in range(len(centroids)):
                    centroids[p][0] = np.mean(df[df['closest'] == p]['x'])
                    centroids[p][1] = np.mean(df[df['closest'] == p]['y'])
                return centroids

            def assignment(df, centroids):
                for p in range(len(centroids)):
                    # sqrt((x1 - x2)^2 + (y1 - y2)^2)
                    df['distance_from_{}'.format(p)] = (
                        np.sqrt(
                            (df['x'] - centroids[p][0]) ** 2
                            + (df['y'] - centroids[p][1]) ** 2
                        )
                    )
                centroid_distance_cols = ['distance_from_{}'.format(p) for p in centroids.keys()]
                df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
                df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
                df['color'] = df['closest'].map(lambda x: colmap[x])
                return df

            def get_numerical_cols(df):
                a = df.describe()
                return df[a.columns], a.columns

            class_col = original_df.columns[::-1][0]
            if original_df[class_col] is not int:
                labels = list(original_df[class_col].unique())
                original_df[class_col] = label_encoder.fit_transform(original_df[class_col])
            numerical_df, numerical_columns = get_numerical_cols(original_df)
            temp_lst = list(numerical_columns)
            if temp_lst.count(class_col) > 0:
                temp_lst.remove(class_col)
                numerical_columns = temp_lst
            list_of_data = frame_to_list(numerical_df)
            X_lst, y_lst = split_data(list_of_data)
            counter = 1
            for i in range(len(numerical_columns)):
                for j in range(len(numerical_columns) - 1 - i):
                    print('Iteration #', counter)
                    counter += 1
                    j = j + i + 1

                    df = pd.DataFrame({
                        'x': X_lst[:, i],
                        'y': X_lst[:, j],
                        'cluster': y_lst
                    })

                    cluster_number = len(df.cluster.unique())

                    centroids = {}
                    for p in range(cluster_number):
                        result_list = []
                        result_list.append(df.loc[df['cluster'] == p]['x'].mean())
                        result_list.append(df.loc[df['cluster'] == p]['y'].mean())
                        centroids[p] = result_list

                    #         Plotting the points
                    fig = plt.figure(figsize=(5, 5))
                    plt.scatter(df['x'], df['y'], c=y_lst)
                    plt.xlabel(numerical_columns[i], fontsize=16)
                    plt.ylabel(numerical_columns[j], fontsize=16)
                    plt.title('Plotting points', fontsize=18)

                    #         Plotting points with centroids
                    fig = plt.figure(figsize=(5, 5))
                    plt.scatter(df['x'], df['y'], c=y_lst, alpha=0.3)
                    col = [i, j]
                    for p in centroids.keys():
                        plt.scatter(centroids[p][0], centroids[p][1], c=colmap[p], edgecolor='k', label=labels[p])
                    plt.xlabel(numerical_columns[i], fontsize=16)
                    plt.ylabel(numerical_columns[j], fontsize=16)
                    plt.title('Plotting points with centroids', fontsize=18)
                    plt.legend()

                    df = assignment(df, centroids)

                    #         Update the centroids
                    centroids = update(df, centroids)
                    while True:
                        closest_centroids = df['closest'].copy(deep=True)
                        centroids = update(df, centroids)
                        df = assignment(df, centroids)
                        if closest_centroids.equals(df['closest']):
                            break

                    #         Final result graph
                    fig = plt.figure(figsize=(5, 5))
                    plt.scatter(df['x'], df['y'], color=df['color'])
                    for p in centroids.keys():
                        plt.scatter(centroids[p][0], centroids[p][1], color=colmap[p], edgecolor='k', label=labels[p])
                    plt.xlabel(numerical_columns[i], fontsize=16)
                    plt.ylabel(numerical_columns[j], fontsize=16)
                    plt.title('Final result', fontsize=18)
                    plt.legend()
                    plt.show()

        df_train = train.copy()
        df_test = test.copy()

        df_train.dropna(inplace=True)
        df_test.dropna(inplace=True)

        sse_range = range(1, 11)
        sse = get_sse(get_numerical_df(df_train)[0])
        kl = KneeLocator(
            sse_range, sse, curve="convex", direction="decreasing"
        )
        plt.figure(figsize=(6, 6))
        plt.plot(sse_range, sse, '-o')
        plt.xlabel('Number of clusters *k*')
        plt.ylabel('Sum of squared distance')
        plt.title('Elbow method')
        plt.show()

        Kmeans_by_me(df_train)

        X_train = dfTrain[dfTrain.columns[:-1]]
        y_train = dfTrain[dfTrain.columns[-1]]
        X_test = dfTest[dfTest.columns[:-1]]
        y_test = dfTest[dfTest.columns[-1]]

        class_col = dfTrain.columns[::-1][0]
        classifiers = len(dfTrain[class_col].unique())

        k_means = KMeans(n_clusters=classifiers)
        k_means.fit(X_train, y_train)
        y_pred = k_means.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        k_means_joblib_model = KMeans(n_clusters=classifiers)
        k_means_joblib_model.fit(X_train, y_train)
        joblib.dump(k_means_joblib_model, 'KMeansJoblib')
        KMeansJoblib = joblib.load('KMeansJoblib')
        jblb_score = KMeansJoblib.score(X_test, y_test)

        print('Accuracy:{0:f}'.format(score * 100))
        print('by joblib Score -> {} %'.format(round(jblb_score * 100, 3)))

        clf = SVC(random_state=0)
        clf.fit(X_train, y_train)
        plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
        plt.show()

        report = classification_report(y_test, y_pred)
        print("Classification report:\n", report)

    def Knn():
        def make_int_list(x):
            temp = []
            for i in range(len(x)):
                temp.append(int(list(x)[i]))
            return temp

        X_train = dfTrain[dfTrain.columns[:-1]]
        y_train = dfTrain[dfTrain.columns[-1]]
        X_test = dfTest[dfTest.columns[:-1]]
        y_test = dfTest[dfTest.columns[-1]]
        knn = KNeighborsRegressor(n_neighbors=1)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)

        knn_model = KNeighborsRegressor(n_neighbors=1)
        knn_model.fit(X_train, y_train)
        joblib.dump(knn_model, 'KNNJoblib')
        KNNJoblib = joblib.load('KNNJoblib')
        jblb_score = KNNJoblib.score(X_test, y_test)

        print("Python accuracy:", accuracy_score(y_test, predictions) * 100)
        print('by joblib Score -> {} %'.format(round(jblb_score * 100, 3)))

        clf = SVC(random_state=0)
        clf.fit(X_train, y_train)
        plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.cool)
        plt.show()

        temp = make_int_list(predictions)
        report = classification_report(y_test, temp)
        print("Classification report:\n", report)

    if alg == "naive bayes":
        my_naive_bayes()
    elif alg == "naive bayes Python":
        Python_naive_bayes()
    elif alg == "Decision tree":
        myDecision_tree()
    elif alg == "Decision tree Python":
        PythonDecision_tree()
    elif alg == "Kmeans":
        Kmeans()
    elif alg == "knn":
        Knn()


fileTrain, fileTest = False, False
global train, test, dfTrain, dfTest
file_window = rootInit()  # create first window
file_frame = Frame(file_window)
myLabel = Label(file_frame,
                text='Please choose your required file: ',
                font=('Arial Bold', 32),
                bg='LightBlue',
                fg='DarkBlue',
                padx=40,
                pady=40
                )
myLabel.pack(side=TOP)
open_train_button = Button(file_frame, text='Click to Open Train File',
                           command=callbackTrain,
                           font=("comic sans", 20)
                           ).pack(side=LEFT)
open_test_button = Button(file_frame, text='Click to Open Test File',
                          command=callbackTest,
                          font=("comic sans", 20)
                          ).pack(side=LEFT)
next_button = Button(file_frame, text='Next',
                     command=startMining,
                     font=("comic sans", 20),
                     state=DISABLED
                     )
next_button.pack(side=LEFT)
cls_button = Button(file_frame, text='Close', font=("comic sans", 20), command=file_window.destroy).pack(side=LEFT)
file_frame.pack(expand=True)
file_window.mainloop()  # show window
