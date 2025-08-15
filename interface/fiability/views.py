from django.shortcuts import render
from django.http import HttpResponse

import pickle
import joblib

import numpy as np

model = joblib.load("./KNeighbors_scoring_clients.pkl")

def home(request):
    if request.method == 'POST': 
        try:
            # Récupérer et convertir les valeurs en float
            MPD = float(request.POST.get('MPD'))
            historique = float(request.POST.get('historique'))
            prix = float(request.POST.get('prix'))
            TV = float(request.POST.get('TV'))
            Solde = float(request.POST.get('Solde'))
            duree = float(request.POST.get('duree'))
            age = float(request.POST.get('age'))
            logement = float(request.POST.get('logement'))

            # Créer la matrice X avec les valeurs converties
            X = np.array([[MPD, historique, prix, TV, Solde, duree, age, logement]])

            # Prédiction
            result = model.predict(X)
            fin = int(result)

            if fin == 0:
                res = "Client non potentiel"
            elif fin == 1:
                res = "Client potentiel"
            else:
                res = "Pas de prédiction"
            
        except ValueError:
            res = "Erreur : Veuillez entrer des valeurs numériques valides."

        context = {'result': res}
        return render(request, 'homepage.html', context)

    return render(request, 'homepage.html')
