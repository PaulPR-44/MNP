# Modélisation Numérique Physique : Le problème à 3 corps

Ce projet vise à simuler et visualiser le problème à 3 corps (et donc techniquement à N corps) selon certains paramètres initiaux.

## Installation
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Démarrage (exemple du système solaire-terre-saturne)
```
python -m three_body.main --config examples/sun_earth_saturn.json
```