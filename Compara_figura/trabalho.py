import sys
import os
import numpy as np
import numpy.random as rnd
import glob
import cv2
import matplotlib.pyplot as plt

imgName=sys.argv[1]; # nome do arquivo -> imagem
img = cv2.imread(imgName)
listaImagens=glob.glob("Archive/*.png") # Varreduras dos arquivos .png do diretorio Archive/
numero_de_imagens = len(listaImagens)
#print (listaImagens)
#print imgName
index = {}
images = {}

def distance(a, b):
   return np.linalg.norm(a-b)

for image_file in glob.glob("Archive/*.png"):
   # Carrega imagem
   image = cv2.imread(image_file)
   images[image_file] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   # Le o histograma
   hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                     [0, 256, 0, 256, 0, 256])
   hist = cv2.normalize(hist).flatten()
   index[image_file] = hist

METHODS = (("Correlacao", cv2.cv.CV_COMP_CORREL),
           ("Qui-Quadrado", cv2.cv.CV_COMP_CHISQR),
           ("Interseccao", cv2.cv.CV_COMP_INTERSECT),
           ("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))

for (method_name, method) in METHODS:
   results = {}
   reverse = False
   if method_name in ("Correlacao", "Interseccao"):
      reverse = True
   for (k, hist) in index.items():
      d = cv2.compareHist(index[imgName], hist, method)
      results[k] = d
   results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)
   # show the query image
   fig = plt.figure("Imagem Buscada")
   ax = fig.add_subplot(1, 1, 1)
   ax.imshow(images[imgName])
   plt.axis("off")

   # initialize the results figure
   fig = plt.figure("Resultados: %s" % (method_name))
   fig.suptitle(method_name, fontsize = 20)

   # loop over the results
   count = 0
   for (i, (v, k)) in enumerate(results):
      # show the result
      ax = fig.add_subplot(1, len(images), i + 1)
      ax.set_title("%s: %.2f" % (k[k.rfind("/")+1:-4], v))
      if method_name == "Correlacao":
         dist = distance(v, 1.0)
         if dist <= 0.01:
            ax.text(0.5, -0.2, "Igual", horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
         elif dist <= 0.3:
            ax.text(0.5, -0.2, "Parecido", horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
         else:
            ax.text(0.5, -0.2, 'Diferente', horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
      elif method_name == "Hellinger":
         dist = distance(v, 0.0)
         if dist <= 0.01:
            ax.text(0.5, -0.2, "Igual", horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
         elif dist <= 0.5:
            ax.text(0.5, -0.2, "Parecido", horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
         else:
            ax.text(0.5, -0.2, 'Diferente', horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
      elif method_name == "Qui-Quadrado":
         dist = distance(v, 0.0)
         if dist <= 1:
            ax.text(0.5, -0.2, "Igual", horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
         elif dist <= 10:
            ax.text(0.5, -0.2, "Parecido", horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
         else:
            ax.text(0.5, -0.2, 'Diferente', horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
      else:
         dist = distance(v, results[0][0]) / distance(results[numero_de_imagens-1][0], results[0][0])
         if dist <= 0.01:
            ax.text(0.5, -0.2, "Igual", horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
         elif dist <= 0.5:
            ax.text(0.5, -0.2, "Parecido", horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
         else:
            ax.text(0.5, -0.2, 'Diferente', horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
      ax.set_yticklabels([])
      ax.set_xticklabels([])
      plt.imshow(images[k])
      plt.axis("off")
      count = count + 1

# show the OpenCV methods
plt.show()










   # histTeste = cv2.calcHist([img],[0],None,[256],[0,256])
# #print histTeste

# correl=[]
# chiquadrado = []
# inters = []
# bhatt = []      

# matrizResultados = [[0.0 for i in range(0,4)]for j in range(0,len(listaImagens))]

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# for i in range(0,len(listaImagens)):
#    imagenslist = []
#    imgcomp=cv2.imread(listaImagens[i])
#    imagenslist.append(imgcomp)             
#    imgcomp = cv2.cvtColor(imgcomp, cv2.COLOR_BGR2RGB)
#    histComp = cv2.calcHist([imgcomp],[0],None,[256],[0,256])

#    matrizResultados[i][0] = cv2.compareHist(histTeste,histComp,cv2.cv.CV_COMP_CORREL) # Correlation Resultado
#    #correl.append(correlacao)
#    print(str(listaImagens[i])+' : '+str(matrizResultados[i][0]))
#    matrizResultados[i][1] = cv2.compareHist(histTeste,histComp,cv2.cv.CV_COMP_CHISQR) # Qui-Quadrado Resultado
#    #quiquadrado.append(quiQuadrado)
#    matrizResultados[i][2]= cv2.compareHist(histTeste,histComp,cv2.cv.CV_COMP_INTERSECT) # Intersection Resultado
#    #inters.append(intersec)
#    matrizResultados[i][3] = cv2.compareHist(histTeste,histComp,cv2.cv.CV_COMP_BHATTACHARYYA) # Batthacaryya Resultado
#    #bhatt.append(battha)
#    #print bhatt

# metodo = ['Correlation','Qui-Quadrado','Intersection','Bhattacharyya'] 
# for j in range(0,4):
#   fig = plt.figure()
#   #fig, ax = plt.subplots(figsize=(5, 3)) ,rotation=45
#   fig.suptitle(metodo[j], fontsize = 20) #  titulo por cada celula do array metodo
#   for k in range (0, len (listaImagens)):
#       ax = fig.add_subplot(1,len(listaImagens), k + 1)
#       image = cv2.imread(listaImagens[k])
#       titulo=listaImagens[k].split('/')
#       #ax.set_title(titulo[1])
#       #ax.set_xlabel("%.2f" % correl[k])
#       #ax.set_xlabel(titulo[1],rotation=45)
#       ax.set_title("%.2f" % matrizResultados[k][j]) #  imprime o resultado das matrizes

#       if (matrizResultados[i][0]):

#         if((matrizResultados[k][j]))==1:
#           ax.set_xlabel(('Exact Match',titulo[1]),rotation=45)
#           #print matrizResultados[k][j]
#         if((matrizResultados[k][j])<1 and (matrizResultados[k][j])>=0.7):
#           ax.set_xlabel(('parecido',titulo[1]),rotation=45,)
#           #print matrizResultados[k][j]
#         if((matrizResultados[k][j])<0.7 and (matrizResultados[k][j])>=-1):
#           ax.set_xlabel(('diferente',titulo[1]),rotation=45,)
#           #print matrizResultados[k][j]
 
#       if (matrizResultados[i][1]):

#         if((matrizResultados[k][j]))==0.0:
#           ax.set_xlabel(('Exact Match',titulo[1]),rotation=45,)
#           #print matrizResultados[k][j]
#         if((matrizResultados[k][j])<0.0 and (matrizResultados[k][j])>=0.67):
#           ax.set_xlabel(('parecido',titulo[1]),rotation=45,)
#           #print matrizResultados[k][j]
#         if((matrizResultados[k][j])<0.67 and (matrizResultados[k][j])>=2.0):
#           ax.set_xlabel(('diferente',titulo[1]),rotation=45,)
#           #print matrizResultados[k][j]

#       if (matrizResultados[i][2]):

#         if((matrizResultados[k][j]))==1:
#           ax.set_xlabel(('Exact Match',titulo[1]),rotation=45,)
#           #print matrizResultados[k][j]
#         if((matrizResultados[k][j])<1 and (matrizResultados[k][j])>=0.5):
#           ax.set_xlabel(('parecido',titulo[1]),rotation=45,)
#           #print matrizResultados[k][j]
#         if((matrizResultados[k][j])<0.5 and (matrizResultados[k][j])>=-0.0):
#           ax.set_xlabel(('diferente',titulo[1]),rotation=45,)
#           #print matrizResultados[k][j]

#       if (matrizResultados[i][3]):

#         if((matrizResultados[k][j]))==0.0:
#           ax.set_xlabel(('Exact Match',titulo[1]),rotation=45,)
#           #print matrizResultados[k][j]
#         if((matrizResultados[k][j])<0.0 and (matrizResultados[k][j])>=0.55):
#           ax.set_xlabel(('parecido',titulo[1]),rotation=45,)
#           #print matrizResultados[k][j]
#         if((matrizResultados[k][j])<0.55 and (matrizResultados[k][j])>=1):
#           ax.set_xlabel(('diferente',titulo[1]),rotation=45,)
#           #print matrizResultados[k][j]
        
#       ax.set_yticklabels([])
#       ax.set_xticklabels([])
#       imgcomp = cv2.imread(listaImagens[k])
#       imgcomp = cv2.cvtColor(imgcomp, cv2.COLOR_BGR2RGB)
#       plt.imshow(imgcomp)
#   #plt.axis("off")


# # show the query imag
# fig = plt.figure("teste")
# ax = fig.add_subplot(1, 1, 1)
# titulo1=imgName.split('/')
# ax.set_title(titulo1[1])
# ax.imshow(img)
# plt.axis("off")

# plt.show()


