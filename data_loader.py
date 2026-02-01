# Charge les datasets mnist

import numpy as np
import csv

def load_data(filename) :
    X = []
    y = []

    # Ouvre le fichier "f" en tant que lecteur, on utilise "with" pour s'assurer que le fichier se ferme automatiquement après utilisation

    with open(filename, 'r') as f : 
        lecteur = csv.reader(f)

        # Passe l'en-tête

        next(lecteur)                

        # Pour chaque ligne du fichier csv, qui contient 60000 lignes, une pour chaque chiffre manuscrit

        for ligne in lecteur :

            # Prend le premier caractère de cette ligne qui est le label (chiffre dont il est question) et l'ajoute à la liste y de labels

            y.append(int(ligne[0]))    

            # Prend toutes les couleurs de pixels du restant de la ligne et divise par 255 car on souhaite une valeur entre 0 et 1

            pixels = np.array(ligne[1:], dtype=np.float32) /255.0       
            
            # Rajoute l'ensemble des pixels à la liste X des entrées
            
            X.append(pixels)        

    # Retourne les deux listes numpy X et y d'entrées et de labels

    return np.array(X), np.array(y)
