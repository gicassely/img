#!/usr/bin/env python
# -*- coding: utf-8 -*-

from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from collections import Counter
import cv2
import numpy as np
import sys
import itertools
import os

def obtem_arquivos(path):
    arquivos = []
    if os.path.isfile(path):
        arquivos.append(path)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file_ in files:
                arquivos.append(os.path.join(root, file_))
    return arquivos

def obtem_nome_pasta(path):
    return os.path.basename(os.path.dirname(path))

def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0]

def obtem_vetor_caracteristica(image, gray_image, mask):
    array = [0] * 8
    contagem = 0
    # Informação de textura com LBP
    padrao = local_binary_pattern(gray_image, 3, 32)
    for linha in padrao:
        for valor in linha:
            if int(valor) < 9:
                array[int(valor)] = array[int(valor)] + 1
                contagem = contagem + 1
    for k in range(8):
        array[k] = float(array[k]) / float(contagem)
    # Informação de forma com Momentos de Hu
    a,im = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(im)
    huMoments = cv2.HuMoments(moments)
    huMoments = list(itertools.chain.from_iterable(huMoments))
    # Informação de cores com média de histogramas normalizados
    histograma_vermelho = list(itertools.chain.from_iterable(cv2.normalize(cv2.calcHist([image], [0], mask, [256], [0, 256]), None)))
    histograma_verde = list(itertools.chain.from_iterable(cv2.normalize(cv2.calcHist([image], [1], mask, [256], [0, 256]), None)))
    histograma_azul = list(itertools.chain.from_iterable(cv2.normalize(cv2.calcHist([image], [2], mask, [256], [0, 256]), None)))
    histograma = histograma_vermelho + histograma_verde + histograma_azul
    array =  array + huMoments + histograma
    return array

def separa_bichos_na_imagem(image):
    height, width, channels = image.shape
    copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 10, 250)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edged, kernel, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    (image, cnts, hiers) = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = cv2.drawContours(copy, cnts, -1, (0, 0, 0), 1, cv2.LINE_AA)
    mask = np.zeros(cont.shape[:2], dtype="uint8") * 255
    cv2.drawContours(mask, cnts, -1, (255, 255, 255), -1)
    image = cv2.bitwise_and(cont, cont, mask=mask)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    elementos = []
    for c in cnts:
        if cv2.contourArea(c) > 0.001 * (width * height):
            x,y,w,h = cv2.boundingRect(c)
            elementos.append([image[y:y+h, x:x+w], gray[y:y+h, x:x+w], mask[y:y+h, x:x+w], arquivo])
    return elementos



arquivos = obtem_arquivos(sys.argv[1])
vetores = []
nomes_bichos = []
for arquivo in arquivos:
    image = cv2.imread(arquivo)
    elementos = separa_bichos_na_imagem(image)    
    for elemento in elementos:
        vetores.append(obtem_vetor_caracteristica(elemento[0], elemento[1], elemento[2]))
        nomes_bichos.append(obtem_nome_pasta(elemento[3]))
kmeans = KMeans(n_clusters=7, random_state=0).fit(vetores)
dicionario = {};
i = 0
for nome in nomes_bichos:
    if nome in dicionario:
        dicionario.get(nome).append(kmeans.labels_[i])
    else:
        dicionario[nome] = [kmeans.labels_[i]]
    i = i + 1
for nome in dicionario:
    dicionario[nome] = most_frequent(dicionario[nome])
print dicionario
if len(sys.argv) > 2:
    arquivos = obtem_arquivos(sys.argv[2])
    i = 0
    for arquivo in arquivos:
        image = cv2.imread(arquivo)
        elementos = separa_bichos_na_imagem(image)
        if not os.path.exists("resposta"):
            os.mkdir("resposta")
        os.chdir("resposta")
        for elemento in elementos:
            vetor = obtem_vetor_caracteristica(elemento[0], elemento[1], elemento[2])
            r = kmeans.predict([vetor])
            nome_especie = "Desconhecido"
            for especie, rotulo in dicionario.items():
                if rotulo == r[0]:
                    nome_especie = especie
            print nome_especie
            if not os.path.exists(nome_especie):
                os.mkdir(nome_especie)
            os.chdir(nome_especie)
            cv2.imwrite(str(i) + ".jpg", elemento[0])
            os.chdir("..")
            print nome_especie
            i = i + 1
        os.chdir("..")


