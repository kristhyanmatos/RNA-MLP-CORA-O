from aprendizagem import Aprendizagem

aprendizagem = Aprendizagem(numero_epocas=600)
taxa_perda, acuracia = aprendizagem.modelo.evaluate(
    aprendizagem.entradas_teste,
    aprendizagem.saidas_teste,
)
aprendizagem.modelo.predict(aprendizagem.entradas_teste)
