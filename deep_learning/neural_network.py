import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Inizializza il modello
model = Sequential()

# Aggiungi layer densi (fully connected)
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Per un problema di classificazione binaria

# Compila il modello
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Addestra il modello
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Valutazione del modello
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
