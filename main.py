from aprendizagem import Aprendizagem

aprendizagem = Aprendizagem(numero_epocas=600)

taxa_perda, acuracia = aprendizagem.modelo.evaluate(
    aprendizagem.entradas_teste,
    aprendizagem.saidas_teste,
)

print("Quantidade de dados de treino: ", len(aprendizagem.entradas_treino))
print("Quantidade de dados de teste: ", len(aprendizagem.entradas_teste))
print("Quantidade de dados de treino e atributos: ", aprendizagem.entradas_treino.shape)
print("Quantidade de dados de teste e atributos: ", aprendizagem.entradas_teste.shape)
print("Quantidade de saídas de treino: ", aprendizagem.saidas_treino.shape)
print("Quantidade de saídas de teste: ", aprendizagem.saidas_teste.shape)
print("Valores possíveis mínimos: ", aprendizagem.saidas_treino.min())
print("Valores possíveis máximos: ", aprendizagem.saidas_treino.max())
