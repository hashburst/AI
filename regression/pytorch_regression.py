import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# Definisci il modello di rete neurale
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x)

# Genera dati di esempio (o carica il tuo dataset)
X = np.random.rand(100, 10)
y = np.random.rand(100, 1)

# Converti i dati in tensori
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

# Inizializza il modello
model = RegressionModel(input_size=X_train.shape[1])

# Definisci l'ottimizzatore e la funzione di perdita
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Addestramento del modello
for epoch in range(100):
    model.train()
    
    # Predizione
    y_pred = model(X_train)
    
    # Calcolo della perdita
    loss = criterion(y_pred, y_train)
    
    # Backpropagation e ottimizzazione
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Valutazione del modello
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    loss = criterion(predictions, y_test)
    print(f'Test Loss: {loss.item()}')
