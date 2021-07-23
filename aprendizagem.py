import pandas
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split


class Aprendizagem:
    def __init__(self, numero_epocas) -> None:
        self.dados = pandas.read_csv("heart.csv")
        self.entradas = self.dados[
            [
                "age",
                "sex",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
            ]
        ].to_numpy()
        self.saidas = self.dados["cp"]

        (
            self.entradas_treino,
            self.entradas_teste,
            self.saidas_treino,
            self.saidas_teste,
        ) = train_test_split(
            self.entradas,
            self.saidas,
            test_size=0.4,
        )

        self.modelo = keras.Sequential(
            [
                keras.layers.Dropout(0.2),
                keras.layers.Dense(180, activation=tensorflow.nn.relu),
                keras.layers.Dense(130, activation=tensorflow.nn.relu),
                keras.layers.Dense(70, activation=tensorflow.nn.relu),
                keras.layers.Dense(40, activation=tensorflow.nn.relu),
                keras.layers.Dense(13, activation=tensorflow.nn.relu),
                keras.layers.Dense(4, activation=tensorflow.nn.softmax),
            ]
        )
        self.modelo.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics="accuracy",
        )

        self.hist = self.modelo.fit(
            self.entradas_treino,
            self.saidas_treino,
            epochs=numero_epocas,
            validation_split=0.4,
        )

        plt.plot(self.hist.history["accuracy"])
        plt.plot(self.hist.history["val_accuracy"])
        plt.title("Acurácia por épocas")
        plt.xlabel("Épocas")
        plt.ylabel("Acurácia")
        plt.legend(["Treino", "Valores de Teste"])
        plt.show()

        plt.plot(self.hist.history["loss"])
        plt.plot(self.hist.history["val_loss"])
        plt.title("Taxa de erro por época")
        plt.xlabel("Épocas")
        plt.ylabel("Taxa de erro")
        plt.legend(["Treino", "Valores de Teste"])
        plt.show()
