import numpy as np
from matplotlib import pyplot as plt

from data_loader import load_data

# On importe les données de nos deux fichiers grâce à l'autre programme "data_loader"

X_train, y_train = load_data("mnist_train.csv")
X_test, y_test = load_data("mnist_test.csv")

def softmax(z):
    z = z - np.max(z)        # stabilité numérique
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

class Couche :

    # Initialisation de la couche entière de neurones

    def __init__(self, n_entrées, n_neurones, est_sortie = False):

        # On définit les poids et les biais de façon aléatoire, comme des listes de dimensions respectives du nombre de neurones x le nombre d'entrées (données) de la couche
        # Les valeurs sont multipliées par 0.01 pour ne pas initialiser le réseau de façon trop brutale

        self.poids = np.random.randn(n_neurones, n_entrées) * 0.01 
        self.biais = np.random.randn(n_neurones) * 0.01
        self.est_sortie = est_sortie

    def calcul(self, a_precédentes) :

        # Le programme calcul la propagation en avant de la couche en faisant le produit des poids par les entrées auxquels on ajoute les biais
        # Comme ces valeurs sont des matrices 2 dimensions, on peut utiliser le produit matriciel pour des calculs plus rapides, résultat auquel on ajoute les biais qui sont une liste (1 dimension)

        self.a_précédentes = a_precédentes
        self.z = np.dot(self.poids, self.a_précédentes) + self.biais

        # Cependant, pour récupérer des valeurs comparables entre elles, on applique une fonction d'activation, qui va convertir les sorties en valeurs comprises entre 0 et 1 

        if self.est_sortie:

            self.a = softmax(self.z)

        else:
            self.a = 1.0 / (1.0 + np.exp(-self.z))
        
        return self.a

    def correction(self, delta, taux_apprentissage) :
        
        # Après la réalisation de la prédiction, l'erreur remonte au fur et à mesure vers le début du réseau, pour corriger les poids et les biais fautifs.
        # Cette correction s'effectue en calculant le delta, puis en utilisant ce delta pour calculer les gradients(dérivées partielles) de la fonction de cout qui calcule la vraisemblabilité par rapport aux poids et biais

        dW = np.outer(delta, self.a_précédentes)
        db = delta

        # Une fois fait, les poids et biais peuvent être mis à jour

        self.poids -= taux_apprentissage * dW
        self.biais -= taux_apprentissage * db
    
class Réseau :

    # Initialisation du réseau entier

    def __init__(self):
        
        # On définit arbitrairement les couches de notre réseau, leur nombre, le nombre de neurones qu'ils possèdent, et le nombre d'entrées qu'ils devront gérer
        # Le nombre d'entrées d'une couche correspond au nombre de sorties de la couche précédente, donc du nombre de neurones 

        self.couche1 = Couche(784, 128)
        self.couche2 = Couche(128, 64)
        self.couche_sortie = Couche(64, 10, est_sortie=True)

    def prediction(self, X):

        # On calcule la prédiction finale de l'entièreté du réseau, grâce à la propagation avant de toutes nos couches

        a = self.couche1.calcul(X)
        a = self.couche2.calcul(a)
        a = self.couche_sortie.calcul(a)

        return a
    
    def entrainement(self, X, y, répétitions = 5, taux_apprentissage = 0.1) :

        # La fonction d'entrainement permet d'entrainer le réseau sur nos données X, dont le résultat est y, sur plusieurs répétitions

        for fois in range(répétitions) :

            total_erreurs = 0

            for xi, cible in zip(X, y):

                # En parcourant nos valeurs d'entrées et nos cibles par couples, on calcule donc les prédictions pour chacune de ces valeurs individuelles
                
                prediction = self.prediction(xi)

                # On calcule ensuite le delta de notre sortie, c'est à dire l'erreur locale du réseau. Autrement dit, en quoi telle ou telle couche a agi sur l'erreur
                # Cependant, notre cible y est une liste de cette forme : (0,0,0,1,0,0,0,0,0,0) ; nous devons donc la convertir en vecteur pour pouvoir effectuer des calculs avec

                cible_vecteur = np.zeros(10)
                cible_vecteur[cible] = 1

                delta_sortie = prediction - cible_vecteur

                # Compteur d'erreurs

                if np.argmax(prediction) != cible :
                    total_erreurs += 1

                # On calcule le reste de nos deltas pour les 2 autres couches du programme

                delta_c2 = (self.couche_sortie.poids.T @ delta_sortie) * (self.couche2.a * (1 - self.couche2.a))
                delta_c1 = (self.couche2.poids.T @ delta_c2) * (self.couche1.a * (1 - self.couche1.a))

                # Puis grâce à ces deltas, on va corriger l'erreur de nos couches en modifiant les biais et poids

                self.couche_sortie.correction(delta_sortie, taux_apprentissage)
                self.couche2.correction(delta_c2, taux_apprentissage)
                self.couche1.correction(delta_c1, taux_apprentissage)

            # Affichage des résultats

            print("Répétition numéro " + str(fois + 1) + " ;") 
            print("Pourcentage d'erreurs : " + str(total_erreurs / len(X) * 100) + " %")

# Comme le réseau est une classe, on initie le réseau sous forme d'objet, duquel on peut appeler la fonction d'entrainement

réseau1 = Réseau()
réseau1.entrainement(X_train, y_train)
