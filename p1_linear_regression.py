# 1) preparar las librerias 
import torch 
import torch.nn as nn 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt  

#2) preparar la data 
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=3784687)

#define X,Y  
x = torch.from_numpy(x_numpy.astype(np.float32)) 
y = torch.from_numpy(y_numpy.astype(np.float32))
    
#la informacion de "y" esta en columnas y queremos que este en vectors, el numero 0 es la cantidad de muestras y el 1 es la cantidad de columnas                     
y=y.view(y.shape[0],1)
    
n_sample,n_feutures=x.shape 

#3) disenar el modelo lineal       
input_size=n_feutures #cantidad de muestras que tengo
output_size=1 #el output/resultado debe ser solo uno por cada muestra  
model = nn.Linear(input_size, output_size)

# 4) "Perdida" y optimizacion 
learning_rate=0.01 #cuan rapido tu modelo aprende 
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate) 

loss_list = []

#5)ciclo/loop
num_epoch = 200 #epoch es cuantas veces el programa va a ir sobre cada conjunto de datos.
for epoch in range(num_epoch): #cada vez que quieras hacer un loop/ciclo puedes hacerlo con "for" 
    optimizer.zero_grad()
    
    y_predicted = model(x)
    loss=criterion(y_predicted,y) #criterion es un tipo de evaluacion o una funcion de costo
    
    loss.backward() #propagar el error devuelta      
    optimizer.step() #luego de propagar el error, se genera una gradiente o un valor numerico que es utilizado para basicamente ir nuevamente sobre la data corrigiendo el error previo. 
    
    loss_list.append(float(loss))
    
    if (epoch+1)%10==0: 
        print('epoch %d: loss = %.2f'%(epoch + 1, loss))
        
#plot
#outpts/resultados
predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro', label='ground truth')
plt.plot(x_numpy,predicted,'b', label='prediction')
plt.title('Final Output')
plt.legend()
plt.show()

#loss/perdida
plt.plot(loss_list)
plt.title('MSE Loss')
plt.show()

        