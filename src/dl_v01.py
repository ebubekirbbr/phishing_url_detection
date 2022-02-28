import os
import json
import time
import pprint
import argparse
import datetime
import numpy as np
import keras.callbacks as ckbs
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, Conv1D, MaxPooling1D, Embedding, Input, GlobalMaxPooling1D, Convolution1D
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from keras_self_attention import SeqSelfAttention
from dl_models import DlModels
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy
# sunucuda calismak icin
#plt.switch_backend('agg')

from keras.models import model_from_json

pp = pprint.PrettyPrinter(indent=4)

TEST_RESULTS = {'data': {},
                "embedding": {},
                "hiperparameter": {},
                "test_result": {}}


class PhishingUrlDetection:

    def __init__(self):

        self.params = {'loss_function': 'categorical_crossentropy',
                       'optimizer': 'adam',
                       'sequence_length': 200,
                       'batch_train': 10000,
                       'batch_test': 10000,
                       'categories': ['phishing', 'legitimate'],
                       'char_index': None,
                       'epoch': 10,
                       'embedding_dimension': 100,
                       'architecture': "cnn",
                       'result_dir': "../result/",
                       'dataset_dir': "../dataset/"}

        self.ml_plotter = Plotter()
        self.dl_models = DlModels(self.params['categories'], self.params['embedding_dimension'], self.params['sequence_length'])

    def set_params(self, args):
        self.params['test_case'] = args.test_case_name
        self.params['epoch'] = int(args.epoch)
        self.params['architecture'] = args.architecture
        self.params['batch_train'] = args.batch_size
        self.params['batch_test'] = args.batch_size

    def model_sum(self, x):
        try:
            TEST_RESULTS['hiperparameter']["model_summary"] += x
        except:
            TEST_RESULTS['hiperparameter']["model_summary"] = x

    def load_data(self, dataset="small_dataset"):
        train = [line.strip() for line in open("{}/{}/train.txt".format(self.params['dataset_dir'], dataset), "r").readlines()]
        test = [line.strip() for line in open("{}/{}/test.txt".format(self.params['dataset_dir'],dataset), "r").readlines()]
        val = [line.strip() for line in open("{}/{}/val.txt".format(self.params['dataset_dir'],dataset), "r").readlines()]

        mlflow.log_param("samples_train", len(train))
        mlflow.log_param("samples_test", len(test))
        mlflow.log_param("samples_val", len(val))
        mlflow.log_param("samples_overall", len(train) + len(test) + len(val))

        mlflow.log_param("dataset", dataset)

        raw_x_train = [line.split("\t")[1] for line in train]
        raw_y_train = [line.split("\t")[0] for line in train]

        raw_x_val = [line.split("\t")[1] for line in val]
        raw_y_val = [line.split("\t")[0] for line in val]

        raw_x_test = [line.split("\t")[1] for line in test]
        raw_y_test = [line.split("\t")[0] for line in test]

        tokener = Tokenizer(lower=True, char_level=True, oov_token='-n-')
        tokener.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
        self.params['char_index'] = tokener.word_index

        x_train = np.asanyarray(tokener.texts_to_sequences(raw_x_train))
        x_val = np.asanyarray(tokener.texts_to_sequences(raw_x_val))
        x_test = np.asanyarray(tokener.texts_to_sequences(raw_x_test))

        encoder = LabelEncoder()
        encoder.fit(self.params['categories'])

        y_train = np_utils.to_categorical(encoder.transform(raw_y_train), num_classes=len(self.params['categories']))
        y_val = np_utils.to_categorical(encoder.transform(raw_y_val), num_classes=len(self.params['categories']))
        y_test = np_utils.to_categorical(encoder.transform(raw_y_test), num_classes=len(self.params['categories']))

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def dl_algorithm(self, x_train, y_train, x_val, y_val, x_test, y_test):

        x_train = sequence.pad_sequences(x_train, maxlen=self.params['sequence_length'])
        x_test = sequence.pad_sequences(x_test, maxlen=self.params['sequence_length'])
        x_val = sequence.pad_sequences(x_val, maxlen=self.params['sequence_length'])

        print("train sequences: {}  |  test sequences: {} | val sequences: {}\n"
              "x_train shape: {}  |  x_test shape: {} | x_val shape: {}\n"
              "Building Model....".format(len(x_train), len(x_test), len(x_val), x_train.shape, x_test.shape, x_val.shape))

        # Build Deep Learning Architecture
        if self.params['architecture'] == "brnn":
            model = self.dl_models.brnn_complex(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "brnn"

        elif self.params['architecture'] == 'att':
            model = self.dl_models.att_complex(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "att"

        elif self.params['architecture'] == 'ann':
            model = self.dl_models.ann_complex(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "ann"

        elif self.params['architecture'] == 'rnn':
            model = self.dl_models.rnn_complex(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "rnn"

        elif self.params['architecture'] == 'cnn':
            model = self.dl_models.cnn_complex(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "cnn"

        else:
            model = self.dl_models.cnn_complex(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "ann"

        # Build Deep Learning Architecture

        model.compile(loss=self.params['loss_function'], optimizer=self.params['optimizer'], metrics=['accuracy'])

        model.summary()
        model.summary(print_fn=lambda x: self.model_sum(x + '\n'))

        hist = model.fit(x_train, y_train,
                         batch_size=self.params['batch_train'],
                         epochs=self.params['epoch'],
                         shuffle=True,
                         validation_data=(x_val, y_val),
                         callbacks=[CustomCallBack()])

        t = time.time()
        score, acc = model.evaluate(x_test, y_test, batch_size=self.params['batch_test'])

        mlflow.log_metric("test_time", float("{0:.2f}".format(round(time.time() - t, 2))))
        TEST_RESULTS['test_result']['test_time'] = time.time() - t

        y_test = list(np.argmax(np.asanyarray(np.squeeze(y_test), dtype=int).tolist(), axis=1))
        y_pred = model.predict_classes(x_test, batch_size=self.params['batch_test'], verbose=1).tolist()
        report = classification_report(y_test, y_pred, target_names=self.params['categories'])
        print(report)
        TEST_RESULTS['test_result']['report'] = report
        TEST_RESULTS['epoch_history'] = hist.history
        TEST_RESULTS['test_result']['test_acc'] = acc
        TEST_RESULTS['test_result']['test_loss'] = score

        test_confusion_matrix = confusion_matrix(y_test, y_pred)
        #TEST_RESULTS['test_result']['y_test'] = y_test
        #TEST_RESULTS['test_result']['y_prediction'] = y_predicted
        TEST_RESULTS['test_result']['test_confusion_matrix'] = test_confusion_matrix.tolist()

        print('Test loss: {0}  |  test accuracy: {1}'.format(score, acc))
        self.save_results(model)

    def traditional_ml(self, x_train, y_train, x_val, y_val, x_test, y_test, algorithm=None):
        print("traditional ML is running")
        if algorithm == "NB":
            gnb = GaussianNB()
            model = gnb.fit(x_train, y_train)
        elif algorithm == "RF":
            clf = RandomForestClassifier(n_estimators=10, random_state=0)
            model = clf.fit(x_train, y_train)
        elif algorithm == 'SVM':
            clf = svm.SVC(gamma='scale')
            model = clf.fit(x_train, y_train)
        else:
            gnb = GaussianNB()
            model = gnb.fit(x_train, y_train)

        model_prediction_val = model.predict(x_val)
        # model_probability = model.predict_proba(x_val)
        acc_val = accuracy_score(y_val, model_prediction_val)

        model_prediction_test = model.predict(x_test)
        acc_test = accuracy_score(y_test, model_prediction_test)
        print("acc_val: {} - acc_test: {}".format(acc_val, acc_test))

    def save_results(self, model):
        tm = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        tsm = tm.split("_")
        TEST_RESULTS['date'] = tsm[0]
        TEST_RESULTS['date_time'] = tsm[1]
        TEST_RESULTS['test_case'] = self.params['test_case']

        TEST_RESULTS['embedding']['vocabulary_size'] = len(self.params['char_index'])
        TEST_RESULTS["embedding"]['embedding_dimension'] = self.params['embedding_dimension']

        TEST_RESULTS['epoch_history']['epoch_time'] = TEST_RESULTS['epoch_times']
        TEST_RESULTS.pop('epoch_times')

        TEST_RESULTS['hiperparameter']['epoch'] = self.params['epoch']
        TEST_RESULTS['hiperparameter']['train_batch_size'] = self.params['batch_train']
        TEST_RESULTS['hiperparameter']['test_batch_size'] = self.params['batch_test']
        TEST_RESULTS['hiperparameter']['sequence_length'] = self.params['sequence_length']

        mlflow.log_param("architecture", TEST_RESULTS['hiperparameter']['architecture'])
        mlflow.log_param("epoch_time_sn", TEST_RESULTS['epoch_history']['epoch_time'])
        mlflow.log_param("val_acc", TEST_RESULTS['epoch_history']['val_acc'])
        mlflow.log_param("train_loss", TEST_RESULTS['epoch_history']['val_loss'])
        mlflow.log_param("train_acc", TEST_RESULTS['epoch_history']['acc'])

        for t in TEST_RESULTS['epoch_history']['epoch_time']:
            mlflow.log_metric("epoch_time", t)

        for t in TEST_RESULTS['epoch_history']['val_loss']:
            mlflow.log_metric("val_loss", t)

        for t in TEST_RESULTS['epoch_history']['val_acc']:
            mlflow.log_metric("val_acc", t)

        for t in TEST_RESULTS['epoch_history']['loss']:
            mlflow.log_metric("train_loss", t)

        for t in TEST_RESULTS['epoch_history']['acc']:
            mlflow.log_metric("train_acc", t)

        mlflow.log_param("epoch_number", self.params['epoch'])
        mlflow.log_param("train_batch_size", self.params['batch_train'])
        mlflow.log_param("test_batch_size", self.params['batch_train'])
        mlflow.log_param("description", self.params['test_case'])
        mlflow.log_param("embed_dim", self.params['embedding_dimension'])

        model_json = model.to_json()
        model.save("{}model_all.h5".format(self.params['result_dir']))
        open("{0}model.json".format(self.params['result_dir']), "w").write(json.dumps(model_json))
        model.save_weights("{0}weights.h5".format(self.params['result_dir']))

        open("{0}raw_test_results.json".format(self.params['result_dir']), "w").write(json.dumps(TEST_RESULTS))
        open("{0}model_summary.txt".format(self.params['result_dir']), "w").write(TEST_RESULTS['hiperparameter']["model_summary"])
        open("{0}classification_report.txt".format(self.params['result_dir']), "w").write(TEST_RESULTS['test_result']['report'])

        self.ml_plotter.plot_graphs(TEST_RESULTS['epoch_history']['acc'], TEST_RESULTS['epoch_history']['val_acc'], save_to=self.params['result_dir'], name="accuracy")
        self.ml_plotter.plot_graphs(TEST_RESULTS['epoch_history']['loss'], TEST_RESULTS['epoch_history']['val_loss'], save_to=self.params['result_dir'], name="loss")
        self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], self.params['categories'], save_to=self.params['result_dir'])
        self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], self.params['categories'], save_to=self.params['result_dir'], normalized=True)

        #TEST_RESULTS['model_json'] = model_json
        mlflow.log_artifact("{0}raw_test_results.json".format(self.params['result_dir']))
        mlflow.log_artifact("{0}model_all.h5".format(self.params['result_dir']))
        mlflow.log_artifact("{0}model_summary.txt".format(self.params['result_dir']))
        mlflow.log_artifact("{0}classification_report.txt".format(self.params['result_dir']))
        mlflow.log_artifact("{0}model.json".format(self.params['result_dir']))
        mlflow.log_artifact("{0}weights.h5".format(self.params['result_dir']))
        mlflow.log_artifact("{0}accuracy.png".format(self.params['result_dir']))
        mlflow.log_artifact("{0}confusion_matrix.png".format(self.params['result_dir']))
        mlflow.log_artifact("{0}normalized_confusion_matrix.png".format(self.params['result_dir']))
        mlflow.log_artifact("{0}loss.png".format(self.params['result_dir']))

        # saving embedding
        embeddings = model.layers[0].get_weights()[0]
        words_embeddings = {w: embeddings[idx].tolist() for w, idx in self.params['char_index'].items()}
        open("{0}char_embeddings.json".format(self.params['result_dir']), "w").write(json.dumps(words_embeddings))


class CustomCallBack(ckbs.Callback):

    def __init__(self):
        ckbs.Callback.__init__(self)
        TEST_RESULTS['epoch_times'] = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        TEST_RESULTS['epoch_times'].append(time.time() - self.epoch_time_start)


class Plotter:

    def plot_graphs(self, train, val, save_to=None, name="accuracy"):

        if name == "accuracy":
            val, = plt.plot(val, label="val_acc")
            train, = plt.plot(train, label="train_acc")
        else:
            val, = plt.plot(val, label="val_loss")
            train, = plt.plot(train, label="train_loss")

        plt.ylabel(name)
        plt.xlabel("epoch")

        plt.legend(handles=[val, train], loc=2)

        if save_to:
            plt.savefig("{0}/{1}.png".format(save_to, name))

        plt.close()

    def plot_graphs_multi(self, values, save_to=None, name="base"):

        train_acc_plotter = []
        val_acc_plotter = []
        train_loss_plotter = []
        val_loss_plotter = []

        # train_acc
        plt.subplot(2, 2, 1)
        plt.ylabel("train_acc")
        plt.xlabel("epoch\na) Train Accuracies")
        plt.ylim(0.75, 1.0)
        plt.xlim(0, 20)
        plt.xticks(range(0, len(values['rnn']['train_acc'])+1, 5))

        p_plot, = plt.plot(values['rnn']['train_acc'], "r", label="rnn")
        train_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['brnn']['train_acc'], "g", label="brnn")
        train_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn']['train_acc'], "b", label="cnn")
        train_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['ann']['train_acc'], "k", label="ann")
        train_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['att']['train_acc'], "c", label="att")
        train_acc_plotter.append(p_plot)

        plt.legend(handles=train_acc_plotter, loc=4)


        # val_acc
        plt.subplot(2, 2, 2)
        plt.ylabel("val_acc")
        plt.xlabel("epoch\nb) Validation Accuracies")
        plt.ylim(0.75, 1.0)
        plt.xlim(0, 20)
        plt.xticks(range(0, len(values['rnn']['train_acc'])+1, 5))

        p_plot, = plt.plot(values['rnn']['val_acc'], "r", label="rnn")
        val_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['brnn']['val_acc'], "g", label="brnn")
        val_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn']['val_acc'], "b", label="cnn")
        val_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['ann']['val_acc'], "k", label="ann")
        val_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['att']['val_acc'], "c", label="att")
        val_acc_plotter.append(p_plot)

        plt.legend(handles=val_acc_plotter, loc=4)


        # train_loss
        plt.subplot(2, 2, 3)
        plt.ylabel("train_loss")
        plt.xlabel("epoch\nc) Train Losses")
        plt.ylim(0.0, 0.6)
        plt.xlim(0, 20)
        plt.xticks(range(0, len(values['rnn']['train_acc'])+1, 5))

        p_plot, = plt.plot(values['rnn']['train_loss'], "r", label="rnn")
        train_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['brnn']['train_loss'], "g", label="brnn") # , linewidth=1.0
        train_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn']['train_loss'], "b", label="cnn")
        train_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['ann']['train_loss'], "k", label="ann")
        train_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['att']['train_loss'], "c", label="att")
        train_loss_plotter.append(p_plot)

        plt.legend(handles=train_loss_plotter, loc=1)


        # val_loss
        plt.subplot(2, 2, 4)
        plt.ylabel("val_loss")
        plt.xlabel("epoch\nd) Validation Losses")
        plt.ylim(0.0, 0.6)
        plt.xlim(0, 20)
        plt.xticks(range(0, len(values['rnn']['train_acc'])+1, 5))

        p_plot, = plt.plot(values['rnn']['val_loss'], "r", label="rnn")
        val_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['brnn']['val_loss'], "g", label="brnn")
        val_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn']['val_loss'], "b", label="cnn")
        val_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['ann']['val_loss'], "k", label="ann")
        val_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['att']['val_loss'], "c", label="att")
        val_loss_plotter.append(p_plot)

        plt.legend(handles=val_loss_plotter, loc=1)
        plt.tight_layout()

        if save_to:
            plt.savefig("{0}/{1}.png".format(save_to, name))

        plt.show()
        #plt.close()

        names = ['rnn', 'brnn', 'cnn', "ann", "att"]
        """
        values = [numpy.mean(values['rnn']['running_time']),
                  numpy.mean(values['brnn']['running_time']),
                  numpy.mean(values['cnn']['running_time']),
                  numpy.mean(values['ann']['running_time']),
                  numpy.mean(values['att']['running_time'])]
        """
        values = [numpy.mean(values['rnn']['running_time']),
                  0,
                  numpy.mean(values['cnn']['running_time']),
                  numpy.mean(values['ann']['running_time']),
                  0]

        print(values)
        plt.ylabel("second")
        plt.xlabel("Architecture")

        plt.bar(names, values)

        plt.show()

    def plot_graphs_multi2(self, values, save_to=None, name="base"):

        train_acc_plotter = []
        val_acc_plotter = []
        train_loss_plotter = []
        val_loss_plotter = []

        # train_acc
        plt.subplot(2, 2, 1)
        plt.ylabel("train_acc")
        plt.xlabel("epoch\na) Train Accuracies")
        plt.ylim(0.8, 1.0)
        plt.xlim(0, 20)
        plt.xticks(range(0, len(values['cnn1']['train_acc'])+1, 10))

        p_plot, = plt.plot(values['cnn1']['train_acc'], "r", label="cnn1")
        train_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn2']['train_acc'], "g", label="cnn2")
        train_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn3']['train_acc'], "b", label="cnn3")
        train_acc_plotter.append(p_plot)

        plt.legend(handles=train_acc_plotter, loc=4)


        # val_acc
        plt.subplot(2, 2, 2)
        plt.ylabel("val_acc")
        plt.xlabel("epoch\nb) Validation Accuracies")
        plt.ylim(0.8, 1.0)
        plt.xlim(0, 20)
        plt.xticks(range(0, len(values['cnn1']['train_acc'])+1, 10))

        p_plot, = plt.plot(values['cnn1']['val_acc'], "r", label="cnn1")
        val_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn2']['val_acc'], "g", label="cnn2")
        val_acc_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn3']['val_acc'], "b", label="cnn3")
        val_acc_plotter.append(p_plot)
        plt.legend(handles=val_acc_plotter, loc=4)


        # train_loss
        plt.subplot(2, 2, 3)
        plt.ylabel("train_loss")
        plt.xlabel("epoch\nc) Train Losses")
        plt.ylim(0.0, 0.5)
        plt.xlim(0, 20)
        plt.xticks(range(0, len(values['cnn1']['train_acc'])+1, 10))

        p_plot, = plt.plot(values['cnn1']['train_loss'], "r", label="cnn1")
        train_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn2']['train_loss'], "g", label="cnn2") # , linewidth=1.0
        train_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn3']['train_loss'], "b", label="cnn3")
        train_loss_plotter.append(p_plot)
        plt.legend(handles=train_loss_plotter, loc=1)


        # val_loss
        plt.subplot(2, 2, 4)
        plt.ylabel("val_loss")
        plt.xlabel("epoch\nd) Validation Losses")
        plt.ylim(0.0, 0.5)
        plt.xlim(0, 20)
        plt.xticks(range(0, len(values['cnn1']['train_acc'])+1, 10))

        p_plot, = plt.plot(values['cnn1']['val_loss'], "r", label="cnn1")
        val_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn2']['val_loss'], "g", label="cnn2")
        val_loss_plotter.append(p_plot)

        p_plot, = plt.plot(values['cnn3']['val_loss'], "b", label="cnn3")
        val_loss_plotter.append(p_plot)
        plt.legend(handles=val_loss_plotter, loc=1)
        plt.tight_layout()

        if save_to:
            plt.savefig("{0}/{1}.png".format(save_to, name))

        plt.show()
        #plt.close()

        names = ['cnn1', 'cnn2', 'cnn3']
        """
        values = [numpy.mean(values['rnn']['running_time']),
                  numpy.mean(values['brnn']['running_time']),
                  numpy.mean(values['cnn']['running_time']),
                  numpy.mean(values['ann']['running_time']),
                  numpy.mean(values['att']['running_time'])]
        """
        values = [numpy.mean(values['cnn1']['running_time']),
                  numpy.mean(values['cnn2']['running_time']),
                  numpy.mean(values['cnn3']['running_time'])]

        print(values)
        plt.ylabel("second")
        plt.xlabel("Architecture")

        plt.bar(names, values)

        plt.show()

    def plot_confusion_matrix(self, confusion_matrix, categories, save_to=None, normalized=False):

        sns.set()
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(14.0, 7.0))

        if normalized:
            row_sums = np.asanyarray(confusion_matrix).sum(axis=1)
            matrix = confusion_matrix / row_sums[:, np.newaxis]
            matrix = [line.tolist() for line in matrix]
            g = sns.heatmap(matrix, annot=True, fmt='f', xticklabels=True, yticklabels=True)

        else:
            matrix = confusion_matrix
            g = sns.heatmap(matrix, annot=True, fmt='d', xticklabels=True, yticklabels=True)

        g.set_yticklabels(categories, rotation=0)
        g.set_xticklabels(categories, rotation=90)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        if save_to:
            if normalized:
                plt.savefig("{0}/{1}.png".format(save_to, "normalized_confusion_matrix"))
            else:
                plt.savefig("{0}/{1}.png".format(save_to, "confusion_matrix"))


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epoch", default=10, help='The number of epoch')
    parser.add_argument("-nm", "--test_case_name", help='Test Case Name')
    parser.add_argument("-arch", "--architecture", default="cnn", help='Architecture to be tested')
    parser.add_argument("-bs", "--batch_size", default=2000, help='batch size')

    args = parser.parse_args()

    return args


def main2():

    pltter = Plotter()

    values = {'ann': {'val_loss': [],
                      'val_acc': [],
                      'train_loss': [],
                      'train_acc': [],
                      'running_time': []},
              'att': {'val_loss': [],
                       'val_acc': [],
                       'train_loss': [],
                       'train_acc': [],
                       'running_time': []},
              'brnn': {'val_loss': [],
                       'val_acc': [],
                       'train_loss': [],
                       'train_acc': [],
                       'running_time': []},
              'cnn': {'val_loss': [],
                       'val_acc':  [],
                       'train_loss': [],
                       'train_acc': [],
                       'running_time': []},
              'rnn': {'val_loss': [],
                       'val_acc': [],
                       'train_loss': [],
                       'train_acc': [],
                       'running_time': []},
              }
    values = {'cnn1': {'val_loss': [
      0.24825197305288643,
      0.14356476816290595,
      0.11373354421559188,
      0.11220497715217374,
      0.09825244803161724,
      0.09049509374785034,
      0.08204772323028,
      0.10174595088854141,
      0.07540850013020281,
      0.08264661463025405,
      0.07113109523147763,
      0.08836916482504663,
      0.07549225745866282,
      0.06629174885726472,
      0.06998371026698715,
      0.08490974329026688,
      0.0768123447605508,
      0.0687867787745274,
      0.06706970183588017,
      0.06637553599175189,
      0.07900188100265483,
      0.060845985987555404,
      0.15649293199628408,
      0.10512269751568903,
      0.0861192455263341,
      0.08366872465142221,
      0.07766995761678024,
      0.07727148507196215,
      0.0760856805286043,
      0.0832283621741689,
      0.07052314350799847,
      0.06851920675950278,
      0.07566765549101365,
      0.0698120651374522,
      0.06676378758792906,
      0.0842661549373686,
      0.06841063969415205,
      0.07445284439823072,
      0.06597370202761403,
      0.061986970574728406,
      0.06074255252156675,
      0.061851741482744235,
      0.08884728451018908,
      0.06524386320968764,
      0.059668090357188695,
      0.06205093943727097,
      0.06089177614361768,
      0.06204384336170525,
      0.06120955924881962,
      0.05741679827161726,
      0.05655363211014982,
      0.0630661090884376,
      0.0724122387327418,
      0.06589350387433514,
      0.05636016875984218,
      0.061555965749701916,
      0.06006044556980672,
      0.06647408157364765,
      0.05580928404228431,
      0.06260230385406557,
      0.05865207926833015,
      0.06368067968184199,
      0.05557045278212368,
      0.05623684510915806,
      0.05452794186153074,
      0.05883555512931175,
      0.10959907494442062,
      0.05483415799763742,
      0.058682701013517205,
      0.060493065321856755,
      0.05571936967815734,
      0.05914123464065674,
      0.05877102069691225,
      0.06958334030773869,
      0.055188295384128026,
      0.05270908780098463,
      0.06204759195444688,
      0.06593447412914771,
      0.06150235104798642,
      0.061528543200437325,
      0.057261605406205104,
      0.05160522773192747,
      0.060341020895286736,
      0.05313695771841964,
      0.08116062203105288,
      0.053349126023759054,
      0.05251808207501479,
      0.05729403389780811,
      0.052988098465211714,
      0.07014045758643661,
      0.05235315431445162,
      0.052724145998895584,
      0.05364312407545827,
      0.06850464860429917,
      0.06827327900228741,
      0.05136922155558206,
      0.0565480775743577,
      0.05365667567187134,
      0.06057837804679096,
      0.05110601020202889
    ],
                       'val_acc': [
      0.8942635037378844,
      0.9420995277647873,
      0.9556016709240589,
      0.9558741919778406,
      0.9609470606269714,
      0.964030943532797,
      0.9676025010773062,
      0.9604498183302299,
      0.9700361491754341,
      0.9676025063301023,
      0.9722402822659459,
      0.9660486153018008,
      0.9696249697480529,
      0.9742579603182423,
      0.972794904684787,
      0.9657952160445684,
      0.9709063214198612,
      0.9742244906595827,
      0.9749655763697304,
      0.9749177621410774,
      0.9695532481748077,
      0.9765290280341227,
      0.9367302276805098,
      0.9592879834827697,
      0.9664406797083749,
      0.9681762564001717,
      0.9699835529097218,
      0.9702943328407797,
      0.9706146701887669,
      0.9688121576404776,
      0.9729909315896808,
      0.973492958481484,
      0.9704042974234794,
      0.9727375317761626,
      0.9739854383889945,
      0.9679993514468577,
      0.9732108684564453,
      0.9700935172142787,
      0.9741862384668544,
      0.9763855988312526,
      0.9769019693811054,
      0.9768589316266549,
      0.9660103574732599,
      0.9744300802615602,
      0.977614360785105,
      0.9767537492178061,
      0.9772079622366145,
      0.9767776582768527,
      0.9772414290636887,
      0.9783937041414047,
      0.9785897365726777,
      0.9762708579886572,
      0.972096841367397,
      0.9751807315870117,
      0.9789913594905136,
      0.9770119446107486,
      0.9768541467580101,
      0.9751377009366031,
      0.9790917581843265,
      0.9761178490262356,
      0.9783124209882069,
      0.9763425659420222,
      0.9794886053620747,
      0.9790678688004668,
      0.9796607281096198,
      0.9782502637353054,
      0.9604354768426066,
      0.9800097546249519,
      0.9783172044251001,
      0.9777291138112691,
      0.9798041623236208,
      0.9782359231231481,
      0.9789100780198522,
      0.9741575481185423,
      0.9793786410255998,
      0.9806026221324794,
      0.9772940187816448,
      0.9762278183373637,
      0.9769449980753945,
      0.9775761141233156,
      0.9792830086879734,
      0.9810185878283391,
      0.9778103967136823,
      0.9804974394774058,
      0.9711692997526874,
      0.9806169689002571,
      0.9805787231230532,
      0.9790295996045468,
      0.9812385222465347,
      0.974300993804796,
      0.9810759691359655,
      0.9813198017610768,
      0.9805787210575005,
      0.9745782957739463,
      0.975424570130638,
      0.981276777165975,
      0.9802870651709333,
      0.9804018167653458,
      0.9785514845988158,
      0.9813580543687396
    ],
                       'train_loss': [
      0.4453441625287232,
      0.18988448030505142,
      0.14238659344902146,
      0.12425194649120838,
      0.11447552415292751,
      0.10218528475227301,
      0.09764431395941754,
      0.09414734168595057,
      0.08998297821044836,
      0.08440917141414801,
      0.08323086485333336,
      0.0805165778784923,
      0.07939771999493173,
      0.07458703655517541,
      0.07268813267057696,
      0.07275805211381933,
      0.07601418886447703,
      0.07146998749681431,
      0.06716951370867853,
      0.06518976081885525,
      0.06472505348814084,
      0.06596159448554322,
      0.3645974258038884,
      0.13476574310659242,
      0.10999244322563193,
      0.10019752466956251,
      0.09431316977567017,
      0.09078252036428516,
      0.08707910939014336,
      0.08493874112756511,
      0.08280057841592801,
      0.07924049535037453,
      0.07923169000110922,
      0.07724389749764626,
      0.07673740740821434,
      0.0736863287428767,
      0.07495446132763997,
      0.07226794903908534,
      0.07085863372834354,
      0.06945417889489691,
      0.06740485234981106,
      0.0674383597126555,
      0.06638530010105162,
      0.07265103960902729,
      0.06683351535469934,
      0.06352487498423592,
      0.06280809119728971,
      0.06245803763231563,
      0.06306095637156638,
      0.06189615069108017,
      0.06065447501142306,
      0.05891641172320967,
      0.06037921899561598,
      0.061040123361005304,
      0.059503055452400595,
      0.05750756813265677,
      0.0581167654959323,
      0.057528860027490886,
      0.05784829169937883,
      0.05613138290448821,
      0.05616301569441963,
      0.05519914415081865,
      0.05642824133826064,
      0.054013721135037975,
      0.05432838545030927,
      0.0537727633123973,
      0.0551510313356843,
      0.06059677811371989,
      0.05267210410372171,
      0.05177838409553906,
      0.05214193108134154,
      0.0521812098512063,
      0.05041875912891857,
      0.050917126842474875,
      0.05100054129357277,
      0.04954824575040643,
      0.049834130951678185,
      0.04976235070619541,
      0.05024261467405491,
      0.04890602439595371,
      0.04832914213498347,
      0.04881872566392193,
      0.048079722264772085,
      0.04881627722433768,
      0.04759801249775385,
      0.051005428176978486,
      0.04675129854456474,
      0.04722811170459053,
      0.04619193016838184,
      0.046528783506250115,
      0.04767182016317429,
      0.046136438687183735,
      0.046138840782965644,
      0.04519616769658328,
      0.04770261599539046,
      0.04508267205120194,
      0.044489262825896915,
      0.04591385668090497,
      0.04323932415504212,
      0.0428956343551715
    ],
                       'train_acc': [
      0.7671259428495503,
      0.9214124685398256,
      0.9426659611084259,
      0.9505888294604307,
      0.9540484726254361,
      0.9595317391811864,
      0.9613727646274871,
      0.9628348781646711,
      0.9645468561136001,
      0.9667516936848648,
      0.9673461418283713,
      0.9681602636985938,
      0.9688288545487347,
      0.9706547805280548,
      0.9715608792773268,
      0.9713796564004294,
      0.9701715278575213,
      0.9720386365874606,
      0.9737876836136935,
      0.9745784548136517,
      0.9747583053675732,
      0.974380763895747,
      0.8413710663405236,
      0.9469864011878567,
      0.9571470570636149,
      0.9608799040963499,
      0.9632687075466675,
      0.9644493749887464,
      0.966363165039887,
      0.9665388990701983,
      0.9673996963749516,
      0.9691226521785848,
      0.969303868856713,
      0.9697006360706208,
      0.969840659085802,
      0.9713645598996145,
      0.9707810858162017,
      0.9716432485376616,
      0.9722651646984136,
      0.9730120083656361,
      0.9736943261555346,
      0.9737917988331006,
      0.9739620376188831,
      0.9716803143011868,
      0.9738494588700188,
      0.9753802207781099,
      0.975554575248911,
      0.9754859321404767,
      0.9752209657387754,
      0.9756355744724301,
      0.9761682498043575,
      0.9770784624688338,
      0.9763686919501102,
      0.9762464998645938,
      0.9767105409785648,
      0.9778088340187805,
      0.9774354124646998,
      0.9775177890317789,
      0.9775823073279013,
      0.9780641882670882,
      0.9779488680245187,
      0.9786339363521129,
      0.9782646296951938,
      0.9788124082908398,
      0.9789524469166873,
      0.9789126262617949,
      0.9784636928303504,
      0.9764538059498464,
      0.9795002159925845,
      0.9799628762052005,
      0.9797528294039528,
      0.9799162033740021,
      0.9804104336674979,
      0.9804639806705187,
      0.9805010452080702,
      0.9804763325152348,
      0.9809582129330024,
      0.98061636615814,
      0.9803651329731946,
      0.9811902319906209,
      0.981195718629276,
      0.9808881959982156,
      0.9812945670341003,
      0.9808428912703591,
      0.9816377837299075,
      0.9802676521227979,
      0.9820784805715571,
      0.9816556342560032,
      0.982074360658885,
      0.981957665001156,
      0.9815293272783987,
      0.9819810034701097,
      0.9822308702462443,
      0.9828239457872444,
      0.9815677727498735,
      0.9825534942879062,
      0.9828747432670949,
      0.982202039377983,
      0.9831712861903341,
      0.9833936939555719
    ],
                       'running_time': [
      17.990022659301758,
      10.97884750366211,
      10.968214273452759,
      10.94959306716919,
      10.93363618850708,
      10.934347152709961,
      10.986653566360474,
      10.989009618759155,
      10.965160131454468,
      10.918111801147461,
      10.971788883209229,
      10.993900537490845,
      10.985745906829834,
      10.965045928955078,
      10.959957361221313,
      10.93389105796814,
      10.945688247680664,
      10.935176610946655,
      10.936874866485596,
      10.938518762588501,
      10.930936574935913,
      10.926815748214722,
      10.952725410461426,
      10.959360361099243,
      10.939948797225952,
      10.95210313796997,
      10.924760580062866,
      10.94145679473877,
      10.959707975387573,
      10.892922163009644,
      10.847553253173828,
      10.861707925796509,
      10.851306200027466,
      10.865909099578857,
      10.85599398612976,
      10.891589641571045,
      10.895450115203857,
      10.880788803100586,
      10.908026218414307,
      10.887088537216187,
      10.909109115600586,
      10.915503025054932,
      10.857759475708008,
      10.862804651260376,
      10.868090867996216,
      10.850837707519531,
      10.864063024520874,
      10.866260051727295,
      10.88959264755249,
      10.901642799377441,
      10.891862392425537,
      10.875759840011597,
      10.911914825439453,
      10.896350860595703,
      10.909618377685547,
      10.924402475357056,
      10.968367576599121,
      10.928229093551636,
      10.98521089553833,
      10.959096670150757,
      10.961496353149414,
      10.945089340209961,
      10.941577672958374,
      10.925239086151123,
      10.923819303512573,
      10.937672853469849,
      10.92611050605774,
      10.958131313323975,
      10.942312955856323,
      10.94141674041748,
      10.95472002029419,
      10.967244625091553,
      10.937745571136475,
      10.951500177383423,
      10.906266689300537,
      10.871194839477539,
      10.853423833847046,
      10.885579824447632,
      10.856873035430908,
      10.860755443572998,
      10.863572120666504,
      10.871357202529907,
      10.885757684707642,
      10.90153193473816,
      10.895807027816772,
      10.93606686592102,
      10.918033361434937,
      10.852494478225708,
      10.878771781921387,
      10.895178318023682,
      10.859812498092651,
      10.86807370185852,
      10.844154357910156,
      10.87431812286377,
      10.880743026733398,
      10.854512453079224,
      10.845232486724854,
      10.871412515640259,
      10.859806299209595,
      10.881233930587769
    ]},
              'cnn2': {'val_loss': [
      0.22084592588615207,
      0.14778606304294695,
      0.16261695091005055,
      0.130438887821955,
      0.17321840662855742,
      0.10237228933999924,
      0.0902554204332958,
      0.08756837584796467,
      0.08472617350363673,
      0.07827546393938792,
      0.11795575972680673,
      0.07469662415226204,
      0.07169721776396128,
      0.15846080178192007,
      0.06941976391019544,
      0.0675951240540367,
      0.06805506701674655,
      0.06625248457366689,
      0.07051189886343368,
      0.06507878025453379,
      0.07015059575281189,
      0.06772348214672694,
      0.0743425660931951,
      0.06162991780574539,
      0.10989191145733035,
      0.2684040066114692,
      0.15461719259300533,
      0.45982599985242034,
      0.14092047893381798,
      0.1050617228805873,
      0.09888826713257787,
      0.09218319634892129,
      0.08809450922267126,
      0.08776581220527156,
      0.13924172188917966,
      0.08752095405379426,
      0.08190120743459942,
      0.07981481723660647,
      0.07709351745622194,
      0.07642726430092622,
      0.07537702654489524,
      0.07481446095276054,
      0.07608662169561191,
      0.07640905605145756,
      0.07310855858007416,
      0.07049690583501937,
      0.07240572959387553,
      0.07216765432797527,
      0.07770086165615603,
      0.11184931659644681,
      0.06781864932947991,
      0.08061469411317856,
      0.07442904702607757,
      0.07050278443344977,
      0.11146175032287763,
      0.07215725778142672,
      0.06592222245927613,
      0.06486848129998013,
      0.06422944477419602,
      0.06406815420816832,
      0.07203949930920349,
      0.0628389846596233,
      0.0636876936767688,
      0.07108281707366064,
      0.06347048451902854,
      0.09453820401912093,
      0.06443809304925564,
      0.08686429983052804,
      0.06150448596506235,
      0.07251106205053236,
      0.07310485032089556,
      0.06565216163856136,
      0.08460784758095752,
      0.06773575564824738,
      0.06084962379081332,
      0.05950607530398789,
      0.059011223557834926,
      0.058615704903189854,
      0.05893922873203856,
      0.08981497850123168,
      0.06092994616707762,
      0.06355400692820869,
      0.05801304499590211,
      0.059727103583061,
      0.05950170253617792,
      0.056967910526792266,
      0.05721817881802514,
      0.06543728353705307,
      0.05649004772851515,
      0.05790803827330552,
      0.07196096089671991,
      0.0551741351903561,
      0.056737250986998056,
      0.057956132793005655,
      0.06800405237757609,
      0.054286002591560106,
      0.1333348975788159,
      0.05620538539480021,
      0.05293475925372948,
      0.06421255560465285
    ],
                       'val_acc': [
      0.9151573953477522,
      0.9418317804904142,
      0.935912642668992,
      0.9503949311911841,
      0.9364290119603623,
      0.9608896867471269,
      0.9643369383170671,
      0.966220741824793,
      0.9668757613324258,
      0.9701174281975266,
      0.9547267059044344,
      0.971532672652781,
      0.9722354990752776,
      0.941783969234698,
      0.9738133050720214,
      0.9742101458160063,
      0.974410955414559,
      0.9749990437211722,
      0.9729239891854319,
      0.9754867299005042,
      0.9735025273113895,
      0.9749464510621975,
      0.9718099811651277,
      0.9766246561084121,
      0.960526316610223,
      0.8926856988628317,
      0.9416070608159977,
      0.8244004376834507,
      0.9455467850161372,
      0.9595174667821009,
      0.9623001442271464,
      0.9642508714807515,
      0.9656948040910932,
      0.9652788394394091,
      0.9450112877543583,
      0.9659577710838986,
      0.9678463505733772,
      0.968606570131411,
      0.9700552745148849,
      0.9704425471630791,
      0.970519042853152,
      0.9708107032857589,
      0.9704616722882232,
      0.9705429582091702,
      0.9718482254057065,
      0.9726992787854835,
      0.9724984677825373,
      0.9718864760572499,
      0.9698879276396595,
      0.9564814143740051,
      0.9736364026311124,
      0.9684392205613918,
      0.9710449808695544,
      0.9728188146466579,
      0.9570216886662722,
      0.9720825014209586,
      0.9745304812352324,
      0.9752237593009612,
      0.9751902913157184,
      0.9752237663685254,
      0.9718530071646231,
      0.9759026979081414,
      0.9752476619125653,
      0.9728714116829627,
      0.9758261999062912,
      0.963624540369871,
      0.9758453145714402,
      0.9661968277318168,
      0.9766676901193333,
      0.97209207080715,
      0.972106408163668,
      0.9753719792909911,
      0.9683961844894775,
      0.97416233326533,
      0.9766676750357834,
      0.9776239373482937,
      0.9772270870562575,
      0.9775665506510796,
      0.9775235318331414,
      0.9673251929798937,
      0.9768541535839091,
      0.9762182531871513,
      0.9775808979477847,
      0.9777912671063346,
      0.9775139587626974,
      0.978489324368418,
      0.9786184137707605,
      0.9752189749202063,
      0.9788192293927018,
      0.9781641989326244,
      0.973856335476205,
      0.9789817968207882,
      0.978785758717225,
      0.9785323614617092,
      0.9741910130487734,
      0.9796655105342553,
      0.9485589430338021,
      0.979373853708386,
      0.980363563282217,
      0.9760365690329229
    ],
                       'train_loss': [
      0.46970714372393313,
      0.19165810792668136,
      0.1444671522585571,
      0.1528134449109317,
      0.11626075840835601,
      0.16044821562829473,
      0.10625515425992826,
      0.09833214707706085,
      0.09552762298316479,
      0.09270885471264499,
      0.08697925396023941,
      0.09009760565345758,
      0.08157421251186057,
      0.0803983293081981,
      0.090524123043074,
      0.07719090286278385,
      0.07475809941035054,
      0.07448083972052237,
      0.07240391141456262,
      0.07202575014967647,
      0.07005342472574794,
      0.06930991991729377,
      0.06684141441718937,
      0.06867231559127371,
      0.06787005842779528,
      0.3895894106304337,
      0.2215616115601629,
      0.14252023326358643,
      0.34423516880760247,
      0.1275893000557088,
      0.11302857230833584,
      0.10638698236807236,
      0.10245222009565418,
      0.09873936310825744,
      0.0974690558166912,
      0.10441814867827028,
      0.09135236102546247,
      0.09029366607771563,
      0.08858916252518276,
      0.08572805880431915,
      0.08501006969597823,
      0.08347999847436742,
      0.08188411963391655,
      0.08055880863234362,
      0.08206890913164357,
      0.079608817081245,
      0.0777450250975921,
      0.07690034581882643,
      0.0769361603829911,
      0.07730696386842384,
      0.08017596944458678,
      0.07253769836217337,
      0.07452743683422036,
      0.07168855502264493,
      0.07217529734842974,
      0.07363962535850639,
      0.07001644560686208,
      0.06981449974240593,
      0.06803550119281931,
      0.0672079790091087,
      0.06729652544070808,
      0.06902907920472853,
      0.06949069789089832,
      0.06607268770768034,
      0.06557332732544427,
      0.06480120327223997,
      0.07148523695623908,
      0.06426169179020441,
      0.0674583506285808,
      0.06301701272748876,
      0.06429310144835097,
      0.06500192743006518,
      0.06162747132151933,
      0.0646676057540948,
      0.06106113873639629,
      0.06104941839979747,
      0.06090191748324631,
      0.06012602684483834,
      0.05957737122177844,
      0.058738493352050815,
      0.0650823029408986,
      0.058405647571442995,
      0.05927625039548714,
      0.05732919347531501,
      0.06082963518621251,
      0.05836820684274385,
      0.05570011321315403,
      0.05656604475972356,
      0.056729137927666186,
      0.055743730838834604,
      0.054910337538839386,
      0.05703065555194962,
      0.05465669100268224,
      0.0549779365452045,
      0.054775486209623823,
      0.05765369107476596,
      0.05353198179386944,
      0.06136919720725842,
      0.05305572746983761,
      0.052615608560367115
    ],
                       'train_acc': [
      0.7468897473890038,
      0.9221606875744189,
      0.9425492636758092,
      0.9401659497042946,
      0.954211849757879,
      0.939067651127149,
      0.9586256374281069,
      0.9612780317114537,
      0.9625342203869856,
      0.9638672821970167,
      0.9660103366562975,
      0.9645770599899163,
      0.968212436636319,
      0.9684595465043099,
      0.9646855175494002,
      0.9698090864392315,
      0.9707165586296015,
      0.9707852041451404,
      0.9716199120427694,
      0.971676201181859,
      0.972598771018661,
      0.9728376543329516,
      0.9739702743621907,
      0.9733552278690228,
      0.9736174446743984,
      0.8441827098055023,
      0.913106570380681,
      0.9446209388148988,
      0.8637956199722091,
      0.9507398417864844,
      0.9560020780364279,
      0.9584334418655286,
      0.9600946225337808,
      0.9612629350846208,
      0.9617846299259434,
      0.9594534863966732,
      0.9641844147430989,
      0.9646127505750974,
      0.9649216495250453,
      0.9664263231716288,
      0.9667832675979966,
      0.9669672357682831,
      0.9680065084536873,
      0.9680875005992582,
      0.9678348895949945,
      0.9686599900701366,
      0.9696498353907351,
      0.969880481757471,
      0.9696635617388635,
      0.9695839370969858,
      0.9684622926688587,
      0.9714949790877453,
      0.9706877273232235,
      0.9715430345575446,
      0.9717064030790402,
      0.9710748813865601,
      0.972432653454827,
      0.972447756670757,
      0.9733840560180808,
      0.9735309521854852,
      0.9735790062250639,
      0.9728678555226326,
      0.9728554993322642,
      0.9739689005713522,
      0.9745853195084404,
      0.9746471024719852,
      0.9719095922879964,
      0.974887353219759,
      0.9739071183203813,
      0.9752209701626566,
      0.9749203067727579,
      0.9745743387708733,
      0.9759115214958276,
      0.9745413913891487,
      0.9763426029709557,
      0.9762602294076839,
      0.9758758268365056,
      0.9764524339715376,
      0.9767064150582778,
      0.9770359052148023,
      0.974748698932117,
      0.9773324534280091,
      0.9769466689633717,
      0.9777058692540056,
      0.9762972995523754,
      0.9771800597574103,
      0.978373089825442,
      0.9780490934984459,
      0.9778225670870972,
      0.9781836307135635,
      0.9783895628088041,
      0.9778596329614201,
      0.9788357430434125,
      0.9787972994250548,
      0.9785845141664775,
      0.977638599465725,
      0.9790911033369768,
      0.9763975183568485,
      0.9796196557147829,
      0.9796512357890422
    ],
                       'running_time': [
      30.473711729049683,
      21.977524995803833,
      21.98611044883728,
      22.00537896156311,
      22.007978200912476,
      22.01161503791809,
      22.010085821151733,
      22.00019645690918,
      22.010628938674927,
      22.02166724205017,
      22.001326322555542,
      21.996913194656372,
      22.003933906555176,
      22.01255226135254,
      22.017797231674194,
      22.01218843460083,
      22.000168800354004,
      22.002402305603027,
      21.990922927856445,
      21.989907264709473,
      22.008313179016113,
      22.021924257278442,
      21.997432470321655,
      21.984650373458862,
      21.982497692108154,
      22.00037908554077,
      21.98266887664795,
      21.980196237564087,
      22.004569053649902,
      22.00984001159668,
      21.988885641098022,
      21.972291231155396,
      21.985946893692017,
      22.003185510635376,
      21.987388610839844,
      21.97954821586609,
      22.003021478652954,
      22.01198649406433,
      21.982657432556152,
      21.994056940078735,
      21.982402086257935,
      21.978180408477783,
      21.97449564933777,
      21.97147798538208,
      21.982210397720337,
      21.97673010826111,
      21.96647024154663,
      21.977136373519897,
      21.970802783966064,
      22.01104760169983,
      21.987494468688965,
      21.96399450302124,
      21.974852800369263,
      21.988856315612793,
      21.981795072555542,
      21.98021912574768,
      21.95620632171631,
      21.96167016029358,
      21.96489977836609,
      21.96265149116516,
      21.983409881591797,
      21.971214056015015,
      21.967418909072876,
      21.9769127368927,
      21.976704835891724,
      21.985129833221436,
      21.97290849685669,
      21.977911233901978,
      21.966432809829712,
      21.98214554786682,
      21.96527099609375,
      21.979830026626587,
      21.973067045211792,
      21.96362614631653,
      21.975656270980835,
      21.969345808029175,
      21.97970962524414,
      21.999077320098877,
      21.973721265792847,
      21.9655704498291,
      21.960994958877563,
      21.961891651153564,
      21.964234352111816,
      21.97290301322937,
      22.00542116165161,
      21.9914767742157,
      22.008960008621216,
      21.982987642288208,
      21.98912525177002,
      21.978187561035156,
      21.96018075942993,
      21.96995711326599,
      21.958513975143433,
      21.974557161331177,
      21.991867780685425,
      22.017328023910522,
      21.993391036987305,
      21.948748350143433,
      21.98314094543457,
      21.98400902748108
    ]},
              'cnn3': {'val_loss': [
      0.47837484966977206,
      0.15999469188733922,
      0.22060704636393227,
      0.10096972894648516,
      0.09777643799159398,
      0.1558516539389329,
      0.09436992352835223,
      0.11509145292393051,
      0.20706756327830192,
      0.07698991708343476,
      0.07833148942943412,
      0.09297806589842536,
      0.06933551658665353,
      0.10332172515037193,
      0.12375712422186662,
      0.06882655708064557,
      0.07028097210388125,
      0.06810200518519234,
      0.06851359217677315,
      0.08512994956915954,
      0.07311650326776442,
      0.07302673148718575,
      0.35461240917024733,
      0.12496713470243546,
      0.10304695567528796,
      0.10189535428577239,
      0.1327860712140999,
      0.08546951317913475,
      0.21626649375872703,
      0.08531300458975018,
      0.08760541317770155,
      0.07527561600729275,
      0.07950545458015934,
      0.09062518331768263,
      0.07198462467082073,
      0.08814886079433133,
      0.08890931499549706,
      0.10479028687061175,
      0.06713767228317487,
      0.08251645493536568,
      0.06523717023920145,
      0.06801046173839466,
      0.06914052380547565,
      0.0677678512188183,
      0.062353440011239346,
      0.07883065558498044,
      0.06248871057301149,
      0.09438705906121464,
      0.06251297111912531,
      0.06223396941560071,
      0.06435545856609594,
      0.0698373775451582,
      0.10781455578252837,
      0.06806344108404125,
      0.06402299322320132,
      0.06265527406118063,
      0.06392712013156471,
      0.06171349115325469,
      0.12618917937423565,
      0.08729262809774119,
      0.06611395777103969,
      0.05939697381849954,
      0.06869303535828493,
      0.08039819016469006,
      0.09192280110993138,
      0.05971164465739393,
      0.06240524106226962,
      0.10537864613737752,
      0.06035105077346908,
      0.16593602708697028,
      0.07294060606988217,
      0.06466829769867341,
      0.06796670253297389,
      0.1945604060741194,
      0.07388765084918593,
      0.07397683982873014,
      0.062049503357524696,
      0.07865301244453307,
      0.06395830086294466,
      0.09821859863102253,
      0.06841215083285519,
      0.09349112227761996,
      0.06988152898414883,
      0.0795830325491389,
      0.07559722248291553,
      0.07510078835017259,
      0.095183069263716,
      0.06805111103345762,
      0.07393833711934863,
      0.12102221296313635,
      0.11139556935576853,
      0.09376166771237843,
      0.18108368177204012,
      0.06705852794048298,
      0.0753996777969482,
      0.07481833532876637,
      0.08439262129760813,
      0.08681011375044376,
      0.08200088575268062,
      0.0862272356202509
    ],
                       'val_acc': [
      0.7870304817276821,
      0.9363190452026766,
      0.9148513936783065,
      0.960650637344602,
      0.9611669956789679,
      0.9485350350599686,
      0.9629121365816751,
      0.9578918723099964,
      0.9243134155577662,
      0.9700648345946901,
      0.9702130534630291,
      0.9641887184319108,
      0.9727996860561274,
      0.9613964978053783,
      0.9580878947007685,
      0.9746500183640087,
      0.9739089297082827,
      0.9743487996709245,
      0.9756492870053871,
      0.9691516145507519,
      0.9755106425617287,
      0.9750659794408733,
      0.8747944149395025,
      0.9539760586736106,
      0.9590776073569695,
      0.9591397633513885,
      0.9470958962097235,
      0.9668709799610854,
      0.9147223054660505,
      0.9660342677953485,
      0.9657999939505225,
      0.9714705129467507,
      0.9697205881922161,
      0.9653983692498365,
      0.9726466906087124,
      0.9666653824069582,
      0.9674303862043956,
      0.9617981176089263,
      0.9750468509096192,
      0.9684439927996153,
      0.9756397310202095,
      0.9748269114529342,
      0.9742005822343371,
      0.9744826776854412,
      0.9767107190917653,
      0.972039484203482,
      0.9768015583030961,
      0.9646620549298047,
      0.9759361620404219,
      0.9772270855150724,
      0.9765816271359801,
      0.9755201936452085,
      0.9653888087824553,
      0.9745304866293799,
      0.9764334210394136,
      0.9775617804784086,
      0.9777817115406512,
      0.9778008326122052,
      0.9553147962720406,
      0.9690942451075139,
      0.9776765273443926,
      0.9784463005439086,
      0.976213481965745,
      0.9732778080245491,
      0.9690894674477846,
      0.9784462916524568,
      0.9788287929743711,
      0.9682288471668631,
      0.979048730857953,
      0.9505909590353799,
      0.9758979140517542,
      0.9781546477397113,
      0.9755919133261703,
      0.9427019521648764,
      0.9731295890467768,
      0.9746978403578628,
      0.9789578865762147,
      0.9736555247514017,
      0.9788526991927128,
      0.970767671199039,
      0.978580173273711,
      0.9727709969526186,
      0.9791443420932007,
      0.9772701119614202,
      0.9765003467094941,
      0.9772940329532509,
      0.9756540760005775,
      0.9784654218936054,
      0.9782311496993976,
      0.9678798178153858,
      0.9708059219873739,
      0.973010060426436,
      0.951810174907764,
      0.9776430452969769,
      0.977839087002718,
      0.9783793724206991,
      0.9789817947552356,
      0.9772510081119242,
      0.9781546496319946,
      0.9777578059469909
    ],
                       'train_loss': [
      0.6544770807567648,
      0.2746135197967877,
      0.15145180009809775,
      0.11832316090580067,
      0.09657816117981005,
      0.12289137695705606,
      0.10535766438941287,
      0.08391022782652356,
      0.07979027179811903,
      0.09522446808426714,
      0.06801532038595962,
      0.06353158853683132,
      0.06167146848217803,
      0.05721009797208204,
      0.06632291369932444,
      0.06318867117812729,
      0.04796917697799248,
      0.048156553647198466,
      0.04631437156541158,
      0.040545872037521954,
      0.03932274639513763,
      0.03929144297305451,
      0.2565552432054827,
      0.2320822362878845,
      0.10828267933065785,
      0.09220119313245692,
      0.08577800177492914,
      0.08729409490961233,
      0.07517461753433188,
      0.09000373646776545,
      0.07215711346678269,
      0.06868379711669463,
      0.06699741230233155,
      0.06586824525119496,
      0.06400328974000961,
      0.06140623061026818,
      0.07184305138190687,
      0.06078490268780765,
      0.06322594085709554,
      0.056356569233792446,
      0.05779798497183977,
      0.05261102287001505,
      0.059640632693172824,
      0.05236575554002233,
      0.05000592074141799,
      0.04901011875251724,
      0.05359642062073952,
      0.048796379011032835,
      0.048725607663212094,
      0.04608278594332828,
      0.05262480503679043,
      0.04607862767361378,
      0.04409916619804278,
      0.05150383072673856,
      0.044401052775118695,
      0.04222053717700268,
      0.04246470772068701,
      0.04042607089972589,
      0.039273673209975726,
      0.04351203055224583,
      0.04182817084170209,
      0.037837712953292975,
      0.0378909435613744,
      0.037469618897662454,
      0.035641188476955604,
      0.04002344438227788,
      0.034867581146984995,
      0.03474858702486001,
      0.0436325245270265,
      0.032584026592862254,
      0.06948809086602754,
      0.03582679743110962,
      0.030618286981199064,
      0.02949882033921129,
      0.09759015151451252,
      0.045075230780890006,
      0.03703252978306866,
      0.03181872082319613,
      0.03159963851991376,
      0.027238121957498095,
      0.030495683418070654,
      0.02498684924437209,
      0.02923738936042047,
      0.024403616877069424,
      0.025067390692084815,
      0.02501478491235112,
      0.021464697112138102,
      0.05477054120963245,
      0.02383611108243508,
      0.022355020425883373,
      0.027284930524486,
      0.027770492196285454,
      0.022010969701861744,
      0.05340151638504148,
      0.024798445942823887,
      0.02027672028216758,
      0.017442504811506564,
      0.016747148821518374,
      0.01675689283520758,
      0.013965286391817874
    ],
                       'train_acc': [
      0.6196351453482998,
      0.8858838131744092,
      0.9398543082583903,
      0.95357345618481,
      0.9620537094469725,
      0.9500877284696048,
      0.9598186710814284,
      0.967866468798094,
      0.9686805848294348,
      0.9629515711683331,
      0.9737561039775509,
      0.9755600655988736,
      0.9758058093050161,
      0.9779474910306992,
      0.9742407340017912,
      0.9754131710101293,
      0.9817434993575697,
      0.9812794614842214,
      0.9820743583625778,
      0.9843684398041163,
      0.984898365502076,
      0.9848599231452061,
      0.8939069005104217,
      0.91591135789651,
      0.9584595212792735,
      0.9642338340509449,
      0.96649496181854,
      0.9656808541236895,
      0.9708099171175063,
      0.9649134165357605,
      0.9722006326599624,
      0.9736737375624374,
      0.9739167314709526,
      0.9746992728715144,
      0.9749903216958414,
      0.9761751137865725,
      0.971806622666297,
      0.9767750624370807,
      0.9755284898275514,
      0.97829620530793,
      0.9775576023799268,
      0.97990933753867,
      0.9770427716417638,
      0.9797596936280567,
      0.9807660140972331,
      0.9810515677146731,
      0.9793135084938086,
      0.9810913827197103,
      0.9811778742266661,
      0.9823681563878294,
      0.9796375041458727,
      0.9824793573875853,
      0.983244052648291,
      0.980260795014284,
      0.9829379017536507,
      0.9839895230004332,
      0.9838371353374492,
      0.9847954042950962,
      0.9851262613260428,
      0.9834238958903947,
      0.9840485552118882,
      0.9857166015619598,
      0.9857756301830515,
      0.9857660224484658,
      0.9864606944821659,
      0.9849258222217391,
      0.9870029876680764,
      0.9869535644577685,
      0.9836147272926808,
      0.9878267113912877,
      0.9747857614956071,
      0.9863357660719675,
      0.9887039812463599,
      0.9892050800668966,
      0.963309892351705,
      0.9829008327169344,
      0.9860749211615982,
      0.9884513707178546,
      0.9883429088503604,
      0.990127653615006,
      0.9887465417460876,
      0.9909994325284862,
      0.9889936576216932,
      0.9912286979562255,
      0.9910213874094499,
      0.9908360563881476,
      0.9925411670715237,
      0.979575725299184,
      0.9915362207509454,
      0.9920002531706664,
      0.9899972280187854,
      0.9897569723764454,
      0.9919316115250213,
      0.9801797938949326,
      0.9908854774658079,
      0.9927512135932409,
      0.9938879544896146,
      0.994087023748257,
      0.9941803777746387,
      0.9952059222486506
    ],
                       'running_time': [
      25.901124715805054,
      17.586766242980957,
      17.577686548233032,
      17.580392599105835,
      17.571945905685425,
      17.58843755722046,
      17.562254667282104,
      17.546008348464966,
      17.557191610336304,
      17.53664469718933,
      17.542457818984985,
      17.578608989715576,
      17.559553384780884,
      17.584449768066406,
      17.607044219970703,
      17.582927703857422,
      17.539390802383423,
      17.544429302215576,
      17.55501127243042,
      17.520856142044067,
      17.5343017578125,
      17.532493352890015,
      17.524407863616943,
      17.543977737426758,
      17.552456617355347,
      17.53853988647461,
      17.51941704750061,
      17.514804363250732,
      17.519553661346436,
      17.524338722229004,
      17.517211198806763,
      17.524826765060425,
      17.5156352519989,
      17.507566213607788,
      17.52781915664673,
      17.515698671340942,
      17.51048231124878,
      17.498252391815186,
      17.492302894592285,
      17.515058517456055,
      17.50236678123474,
      17.501954555511475,
      17.524008750915527,
      17.504855155944824,
      17.49942898750305,
      17.49780774116516,
      17.49227285385132,
      17.475778341293335,
      17.504262685775757,
      17.509891510009766,
      17.49823784828186,
      17.491461992263794,
      17.484268188476562,
      17.49100613594055,
      17.51883840560913,
      17.548537015914917,
      17.542258977890015,
      17.561439990997314,
      17.578720808029175,
      17.58121156692505,
      17.56749725341797,
      17.553659200668335,
      17.54105496406555,
      17.552074909210205,
      17.573769330978394,
      17.546945571899414,
      17.551878213882446,
      17.536414861679077,
      17.56895422935486,
      17.475687265396118,
      17.472904205322266,
      17.479534149169922,
      17.470067024230957,
      17.50939679145813,
      17.513317346572876,
      17.537174463272095,
      17.5027072429657,
      17.49965000152588,
      17.466726779937744,
      17.486379623413086,
      17.454399824142456,
      17.460741758346558,
      17.459854125976562,
      17.470375537872314,
      17.451029062271118,
      17.457164764404297,
      17.475170373916626,
      17.461254835128784,
      17.45699381828308,
      17.452878952026367,
      17.48060178756714,
      17.459007263183594,
      17.45131015777588,
      17.48957872390747,
      17.486811637878418,
      17.499382972717285,
      17.501418113708496,
      17.520119667053223,
      17.51648211479187,
      17.49273180961609
    ]}}

    pltter.plot_graphs_multi2(values, name="train_acc")



def main():

    args = argument_parsing()
    vc = PhishingUrlDetection()
    vc.set_params(args)

    mlflow.start_run(experiment_id=1)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = vc.load_data(dataset="small_dataset")
    #vc.dl_algorithm(x_train, y_train, x_val, y_val, x_test, y_test)
    vc.traditional_ml(x_train, y_train, x_val, y_val, x_test, y_test)
    mlflow.end_run()


if __name__ == '__main__':
    main()

