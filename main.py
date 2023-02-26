import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# définir les caractères à conserver
caracteres = "abcdefghijklmnopqrstuvwxyz "

# définir la chaîne de texte
texte = "eeeee Ceci est une phrase de test avec des lettres minuscules et des espaces"

# initialiser un Tokenizer Keras avec les caractères à conserver
tokenizer = Tokenizer(char_level=True, filters=caracteres)

text = texte.lower()

# ajuster le Tokenizer sur le texte
tokenizer.fit_on_texts([text])

# convertir la chaîne de texte en représentation "one-hot encoding" des caractères
one_hot = tokenizer.texts_to_sequences([texte])
print(len(one_hot[0]))
one_hot = pad_sequences(one_hot, maxlen=10)
# afficher la représentation "one-hot encoding"
print(one_hot[0])
print(len(one_hot[0]))