# Modélisation Numérique Physique : Le problème à 3 corps

Ce projet vise à simuler et visualiser le problème à N corps (typiquement 3 corps) en utilisant des méthodes d'intégration numérique. Il permet d'explorer la dynamique orbitale et la recherche de conditions initiales stables dans un système intrinsèquement chaotique.

## 1. Introduction et Problématique

### Contexte Historique
Le problème à 3 corps a marqué l'histoire des sciences :
*   **Isaac Newton (XVIIe siècle)** : Pose les bases avec la loi de la gravitation universelle.
*   **Euler et Lagrange (XVIIIe siècle)** : Découvrent les premières solutions particulières (points de Lagrange, configurations alignées ou triangulaires).
*   **Henri Poincaré (Fin XIXe siècle)** : Démontre que le problème n'est pas intégrable et met en évidence la sensibilité aux conditions initiales, posant les jalons de la **théorie du chaos**.

### La Question de la Stabilité
Comment trouver des conditions initiales permettant la stabilité du problème à 3 corps ?
Contrairement au problème à 2 corps, le système à 3 corps est souvent instable. La stabilité repose sur trois piliers :
1.  **Hiérarchie** : Un corps massif domine le système (ex: Soleil-Terre-Lune).
2.  **Symétrie et Périodicité** : Solutions mathématiques comme la **Figure en 8** ou le **Triangle équilatéral**.
3.  **Recherche Numérique** : Utilisation d'indicateurs de chaos pour isoler des "îlots de stabilité".

## 2. Modélisation Physique

### Loi de la Gravitation
Le système résout l'accélération $\mathbf{a}_i$ pour chaque corps $i$ :
$$\mathbf{a}_i = G \sum_{j \neq i} m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{(|\mathbf{r}_j - \mathbf{r}_i|^2 + \epsilon^2)^{3/2}}$$
L'utilisation du **softening de Plummer** ($\epsilon$) permet d'éviter les singularités numériques lors de rencontres très proches.

### Intégration Numérique
Nous utilisons l'algorithme de **Runge-Kutta d'ordre 4 (RK4)**, qui offre un excellent compromis entre précision locale et coût de calcul pour les systèmes hiérarchiques.

## 3. Architecture et Choix Techniques

Le projet est conçu de manière modulaire :
*   `physics.py` : Calcul des accélérations et gestion du vecteur d'état.
*   `integrators.py` : Implémentation générique du RK4.
*   `simulate.py` : Orchestration de la simulation.
*   `visualize.py` : Animation 2D/3D avec Matplotlib.
*   `main.py` : Interface en ligne de commande (CLI).

**Choix technologiques :**
*   **NumPy** : Pour les calculs vectorisés performants.
*   **Matplotlib** : Pour les visualisations et animations.

## 4. Installation

```bash
# Création de l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Installation des dépendances
pip install -r requirements.txt
```

## 5. Utilisation

### Ligne de commande
Pour lancer l'exemple par défaut (Soleil-Terre-Saturne) :
```bash
python -m three_body.main --config examples/sun_earth_saturn.json
```

Options disponibles :
*   `--config` : Chemin vers un fichier JSON de configuration.
*   `--trail` : Nombre de points pour la trace des trajectoires.
*   `--save` : Sauvegarder l'animation (ex: `output.gif` ou `output.mp4`).

### Rapport Interactif
Un rapport technique détaillé et auto-contenu est disponible sous forme de calepin Jupyter : `rapport_technique.ipynb`. Il inclut les explications théoriques, le code source intégré et une démonstration interactive.

## 6. Perspectives d'Amélioration

*   **Intégrateurs Symplectiques** (ex: Yoshida) : Pour une meilleure conservation de l'énergie sur le long terme.
*   **Pas de temps adaptatif** (ex: Dormand-Prince) : Pour optimiser les calculs lors de rencontres proches.
*   **Radau IIA** : Pour gérer les systèmes à forte raideur numérique.

---
Projet développé par [Nicolas Fortun](https://github.com/NFortun) et [Paul Peron Redon](https://github.com/PaulPR-44) dans le cadre de l'UE Modélisation numérique en physique de Sorbonne Université.