import time
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay,matthews_corrcoef,f1_score, recall_score,accuracy_score,precision_score

def calculate_metrics(y_test, Y_predicted):
    from sklearn import metrics
    from sklearn.metrics import classification_report, confusion_matrix
    cm = confusion_matrix(y_test,Y_predicted)
    cm_plot=ConfusionMatrixDisplay(cm)
    cm_plot.plot()
    plt.savefig('cm.png')

    accuracy_values = []
    accuracy = metrics.accuracy_score(y_test, Y_predicted)
    accuracy_values.append(accuracy)
    print("Accuracy = " + str(round(accuracy * 100, 2)) + "%")

    confusion_mat = confusion_matrix(y_test, Y_predicted)
    print(confusion_mat)
    print(confusion_mat.shape)

    print("TP\tFP\tFN\tTN\tSensitivity\tSpecificity\tPrecision")
    sensitivity_values = []
    specificity_values = []
    precision_values = []
    for i in range(confusion_mat.shape[0]):
        TP = round(float(confusion_mat[i, i]), 2)
        FP = round(float(confusion_mat[:, i].sum()), 2) - TP
        FN = round(float(confusion_mat[i, :].sum()), 2) - TP
        TN = round(float(confusion_mat.sum().sum()), 2) - TP - FP - FN
        print(str(TP) + "\t" + str(FP) + "\t" + str(FN) + "\t" + str(TN))
        sensitivity = round(TP / (TP + FN), 2)
        sensitivity_values.append(sensitivity)
        specificity = round(TN / (TN + FP), 2)
        specificity_values.append(specificity)
        precision = round(TP/(TP+FP),2)
        precision_values.append(precision)
        print("\t" + str(sensitivity) + "\t\t" + str(specificity) + "\t\t" + str(precision))


     # Plotting accuracy values
    classes = range(confusion_mat.shape[0])
    plt.bar(classes, accuracy_values)
    plt.xlabel("Class")
    plt.ylabel("accuracy")
    plt.title("accuracy for Each Class")
    plt.xticks(classes)
    plt.show()

    # Plotting sensitivity values
    classes = range(confusion_mat.shape[0])
    plt.bar(classes, sensitivity_values)
    plt.xlabel("Class")
    plt.ylabel("Sensitivity")
    plt.title("Sensitivity for Each Class")
    plt.xticks(classes)
    plt.show()

    #Plotting specificity values
    classes = range(confusion_mat.shape[0])
    plt.bar(classes, specificity_values)
    plt.xlabel("Class")
    plt.ylabel("Specificity")
    plt.title("Specificity for Each Class")
    plt.xticks(classes)
    plt.show()

    
     # Plotting precision values
    classes = range(confusion_mat.shape[0])
    plt.bar(classes, precision_values)
    plt.xlabel("Class")
    plt.ylabel("precision")
    plt.title("precision for Each Class")
    plt.xticks(classes)
    plt.show()

    fscore_values = []
    f_score = metrics.f1_score(y_test, Y_predicted)
    fscore_values.append(f_score)
    print("F1-score = " + str(round(f_score, 2)))

    #Plotting fscore values
    classes = range(confusion_mat.shape[0])
    plt.bar(classes, specificity_values)
    plt.xlabel("Class")
    plt.ylabel("fscore")
    plt.title("fscore for Each Class")
    plt.xticks(classes)
    plt.show()

def neural_network(dataset, Target_labels, test_size):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    X = pd.read_csv("C:/Users/SATHI/Onedrive/Desktop/Malicious-Website-detection-using-Machine-Learning-master/ML Algorithm Evaluation/Dataset.csv")
    Y = pd.read_csv("C:/Users/SATHI/Onedrive/Desktop/Malicious-Website-detection-using-Machine-Learning-master/ML Algorithm Evaluation/Target_labels.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', random_state=42)
    model.fit(X_train, y_train)
    Y_predicted = model.predict(X_test)

    return y_test, Y_predicted


def random_forests(dataset, Target_labels, test_size):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X = pd.read_csv("C:/Users/SATHI/Onedrive/Desktop/Malicious-Website-detection-using-Machine-Learning-master/ML Algorithm Evaluation/Dataset.csv")
    Y = pd.read_csv("C:/Users/SATHI/Onedrive/Desktop/Malicious-Website-detection-using-Machine-Learning-master/ML Algorithm Evaluation/Target_labels.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    model = RandomForestClassifier(n_estimators=5, criterion='entropy', random_state=42)
    model.fit(X_train, y_train)
    Y_predicted = model.predict(X_test)

    return y_test, Y_predicted


def support_vector_machines(dataset, Target_labels, test_size):
    import pandas as pd
    from sklearn import svm
    from sklearn.model_selection import train_test_split

    X = pd.read_csv("C:/Users/SATHI/Onedrive/Desktop/Malicious-Website-detection-using-Machine-Learning-master/ML Algorithm Evaluation/Dataset.csv")
    Y = pd.read_csv("C:/Users/SATHI/Onedrive/Desktop/Malicious-Website-detection-using-Machine-Learning-master/ML Algorithm Evaluation/Target_labels.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    model = svm.SVC(kernel='rbf', C=2.0)
    model.fit(X_train, y_train)
    Y_predicted = model.predict(X_test)

    return y_test, Y_predicted


def main():
    dataset = "Dataset.csv"
    class_labels = "Target_Labels.csv"
    test_size = 0.3

    print("\nRunning neural networks...")
    start_time = time.time()
    y_test, Y_predicted = neural_network(dataset, class_labels, test_size)
    calculate_metrics(y_test, Y_predicted)
    end_time = time.time()
    print("Runtime = " + str(end_time - start_time) + " seconds")

    print("\nRunning random forests...")
    start_time = time.time()
    y_test, Y_predicted = random_forests(dataset, class_labels, test_size)
    calculate_metrics(y_test, Y_predicted)
    end_time = time.time()
    print("Runtime = " + str(end_time - start_time) + " seconds")

    print("\nRunning support vector machines...")
    start_time = time.time()
    y_test, Y_predicted = support_vector_machines(dataset, class_labels, test_size)
    calculate_metrics(y_test, Y_predicted)
    end_time = time.time()
    print("Runtime = " + str(end_time - start_time) + " seconds")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total runtime = " + str(end_time - start_time) + " seconds")


    