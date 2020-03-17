from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys


def plot_metrics(type, x, al, knn, svm, rf, mlp):

    plt.title(type)
    plt.xlabel("Digit Number")
    plt.ylabel("Accuracy")
    plt.plot(x, knn, label='KNN')
    plt.plot(x, svm, label='SVM')
    plt.plot(x, rf, label='RF')
    plt.plot(x, mlp, label='MLP')
    
    leg = plt.legend(loc='best', ncol=4, mode="expand", shadow=True)
    leg.get_frame().set_alpha(al)
    

if __name__ == '__main__':

    # Load dataset
    mnist = datasets.load_digits()

    # Split data
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
    mnist.target, test_size=0.25, random_state=42)

    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
    test_size=0.1, random_state=84)

    print("the number of train data is:" + str(len(trainData)))
    print("the number of test data is:" + str(len(testData)))

    #sys.exit()
    
    # KNN
    kVals = range(1, 30, 2)
    accuracies = []

    for k in range(1, 30, 2):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(trainData, trainLabels)
            score = model.score(valData, valLabels)
            print("k=%d, accuracy=%.2f%%" % (k, score * 100))
            accuracies.append(score)

    i = np.argmax(accuracies)
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
    accuracies[i] * 100))

    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)

    print("EVALUATION ON TESTING DATA")
    print(classification_report(testLabels, predictions))
    knn_Precision, knn_Recall, knn_Fscore, support = metrics.precision_recall_fscore_support(testLabels, predictions)
    knn_Acc = metrics.accuracy_score(testLabels, predictions)

    print ("Confusion matrix")
    print(confusion_matrix(testLabels,predictions))

    # SVM
    clf = SVC(kernel='rbf', gamma=0.001)
    clf.fit(trainData, trainLabels)
    predictions = clf.predict(testData)
    
    print("SVM Results")
    print(classification_report(predictions, testLabels))
    svm_Precision, svm_Recall, svm_Fscore, support = metrics.precision_recall_fscore_support(testLabels, predictions)
    svm_Acc = metrics.accuracy_score(testLabels, predictions)
    print("SVM Accuracy is:" + str(svm_Acc))
    print ("Confusion matrix")
    print(confusion_matrix(testLabels,predictions))

    # Random Forest
    print("Below Random Forests")
    accuracies = {}
    best_acc = 0
    best_k = 0
    best_f = 0
    max_fea = [a for a in range(5, 16, 2)]
    max_k = [10]
    max_k.extend([a for a in range(50, 201, 50)])
    print(type(max_fea))
    max_fea.append(20)
    max_fea.append(30)
    max_fea.append(50)
    
    for k in max_k:
        
        scores = []
        for j in max_fea:
            RF_model = RandomForestClassifier(n_estimators=k, max_features=j)
            RF_model.fit(trainData, trainLabels)
            score = RF_model.score(valData, valLabels)
            print("k=%d, max_features=%d, accuracy=%.2f%%" % (k, j, score * 100))
            scores.append(score)
            
            if best_acc < score:
                best_acc = score
                best_k = k
                best_f = j
        accuracies[k] = scores
    
    print("k=%d, max_features=%d achieved highest accuracy of %.2f%% on validation data" % (best_k, best_f,
    best_acc * 100))
    
    RF_model = RandomForestClassifier(n_estimators=best_k, max_features=best_f)
    RF_model.fit(trainData, trainLabels)
    predictions = RF_model.predict(testData)
    print("Random Forests Results")
    print(classification_report(predictions, testLabels))
    rf_Precision, rf_Recall, rf_Fscore, support = metrics.precision_recall_fscore_support(testLabels, predictions)
    rf_Acc = metrics.accuracy_score(testLabels, predictions)
    print("When k=%d, Accuracy=%.2f%%" % (best_k, rf_Acc * 100))

    plt.title("Validation Accuracy of RF on Max_Features")
    for k in max_k:
        plt.plot(max_fea, accuracies[k], label=k)
        
    leg = plt.legend(loc='best', ncol=5, shadow=True)
    plt.xlim([5, 50])
    plt.xlabel("The value of max_features")
    plt.ylabel("Accuracy")
    plt.savefig("Validation Accuracy of RF on Max_Features")
    plt.show()

    #sys.exit()

    # MLP
    # Alpha is the regularization term, smaller alpha, less underfitting 
    mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(200, 200), alpha = 0.4, max_iter=400)
    mlp.fit(trainData, trainLabels)
    predictions = mlp.predict(testData)

    print("Multi-Layer Perceptron")
    print(classification_report(predictions, testLabels))
    mlp_Precision, mlp_Recall, mlp_Fscore, support = metrics.precision_recall_fscore_support(testLabels, predictions)
    mlp_Acc = metrics.accuracy_score(testLabels, predictions)
    print(mlp_Acc)

    #sys.exit()
    
    # Draw picture
    x = [a for a in range(0, 10)]
    plt.figure(figsize=(10, 6.5))
    
    plt.subplot(221)
    alpha = 0.98 / 4 * 3 + 0.01
    plot_metrics('Precision', x, alpha, knn_Precision, svm_Precision, rf_Precision, mlp_Precision)

    alpha = 0.98 / 4 * 3 + 0.01
    plt.subplot(222)
    plot_metrics('Recall', x, alpha, knn_Recall, svm_Recall, rf_Recall, mlp_Recall)

    alpha = 0.98 / 4 * 3 + 0.01
    plt.subplot(223)
    plot_metrics('Fscore', x, alpha, knn_Fscore, svm_Fscore, rf_Fscore, mlp_Fscore)

    plt.subplot(224)
    plt.title("Accuracy")
    colors = ['b','g','r','m']
    Label_Com = ['KNN','SVM','RF','MLP']
    y_vals = [knn_Acc, svm_Acc, rf_Acc, mlp_Acc]
    
    for index in range(4):
        plt.scatter(index, y_vals[index] , c=colors[index], cmap='brg', s=40, alpha=0.2, marker='8', linewidth=0)  
    plt.legend(["KNN", "SVM", "RF", "MLP"], loc='best')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.5, hspace=0.5)
    plt.savefig("Performance") # save then show
    plt.show()
    