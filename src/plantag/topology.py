"""Local topology of the expression latent space (verbatim from the PlantAG notebook, cells 43-48)."""
import numpy as np
import gudhi as gd
from persim import PersistenceImager, wasserstein, bottleneck
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler


def compute_local_diagrams(X, n_neighbors=30, max_dim=2):
    """
    Calcule les diagrammes de persistance locaux pour chaque point de X.

    X : ndarray, shape (n_samples, n_features)
        Les données à analyser.
    n_neighbors : int
        Nombre de voisins pour construire le diagramme local autour de chaque point.
    max_dim : int
        Dimension maximale des complexes simpliciaux.

    Retour :
    diagrams : list
        Liste des diagrammes de persistance pour chaque point.
    """

    # Construction du voisinage pour chaque point
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)  # distances et indices des voisins

    diagrams = []  # Liste pour stocker les diagrammes

    # Boucle sur chaque point pour calculer le diagramme local
    for i in range(X.shape[0]):
        neighbors_idx = indices[i]       # indices des voisins du point i
        points = X[neighbors_idx, :]    # coordonnées des voisins
        rips = gd.RipsComplex(points=points, max_edge_length=np.max(distances[i]))  # complexe de Rips
        st = rips.create_simplex_tree(max_dimension=max_dim)  # création de l'arbre simplicial
        st.compute_persistence()         # calcul de la persistance
        diagrams.append(st.persistence())  # ajout du diagramme à la liste

    return diagrams  # retourne la liste des diagrammes


def diagrams_to_images(diagrams_clean, pixel_size=0.1):
    """
    Transforme une liste de diagrammes de persistance en images (grilles de pixels).

    diagrams_clean : list
        Liste des diagrammes de persistance nettoyés.
    pixel_size : float
        Taille d'un pixel dans l'image de persistance.

    Retour :
    images : list
        Liste des images normalisées correspondant aux diagrammes.
    """

    # Création de l'objet PersistenceImager avec la taille de pixel choisie
    pimgr = PersistenceImager(pixel_size=pixel_size)

    # Apprentissage de la “grille” pour transformer les diagrammes en images
    pimgr.fit(diagrams_clean)

    # Transformation de chaque diagramme en image
    images = [pimgr.transform(d) for d in diagrams_clean]

    # Normalisation des images pour que les valeurs soient entre 0 et 1
    # Ajout de 1e-8 pour éviter la division par zéro
    images = [img / (img.max() + 1e-8) for img in images]

    return images  # retourne la liste d'images normalisées


def diagrams_birth_to_death(diagrams_clean):
    """
    Convertit des diagrammes de persistance de format (birth, lifetime)
    en format (birth, death).

    diagrams_clean : list
        Liste de diagrammes de persistance (birth, lifetime).

    Retour :
    diagrams_bd : list
        Liste de diagrammes de persistance convertis en (birth, death).
    """

    diagrams_bd = []  # Liste pour stocker les diagrammes convertis

    # Boucle sur chaque diagramme
    for diag in diagrams_clean:
        # Cas particulier : diagramme vide → ajout d'un point artificiel
        if len(diag) == 0:
            diagrams_bd.append(np.array([[0.0, 1e-6]]))
            continue

        # Extraction des naissances et durées
        births = diag[:, 0]                 # colonnes de naissance
        deaths = diag[:, 0] + diag[:, 1]    # calcul des décès = naissance + durée

        # S'assure que death > birth pour tous les points
        deaths = np.maximum(deaths, births + 1e-8)

        # Création du diagramme (birth, death) et ajout à la liste
        diagrams_bd.append(np.column_stack([births, deaths]))

    return diagrams_bd  # retourne la liste des diagrammes convertis


def clean_diagrams(diagrams):
    """
    Nettoie les diagrammes de persistance : remplace les valeurs infinies et convertit birth→death.
    """
    cleaned = []
    for diag in diagrams:
        # Récupère les valeurs finies de death pour calculer un max
        finite_vals = [d for _, (b, d) in diag if np.isfinite(d)]
        max_val = max(finite_vals) if finite_vals else 1.0
        points = []
        for _, (b, d) in diag:
            if not np.isfinite(d):
                d = b + max_val  # remplace l'infini par une valeur finie
            if d > b:
                points.append([b, d-b])  # conversion en (birth, lifetime)
        cleaned.append(np.array(points))
    return cleaned


def superlevel_filtration(img, n_levels=5, mode="quantile"):
    """
    Crée des images binaires filtrées par seuils adaptatifs ou linéaires.
    Chaque niveau correspond à un seuil pour extraire la topologie.
    """
    if mode == "linear":
        thresholds = np.linspace(0, 1, n_levels)
    elif mode == "quantile":
        thresholds = np.quantile(img, np.linspace(0, 1, n_levels))
    else:
        raise ValueError("mode doit être 'linear' ou 'quantile'")
    return [(t, (img >= t).astype(int)) for t in thresholds]


def persistence_from_image(img):
    """
    Calcule le diagramme de persistance d'une image via CubicalComplex.
    """
    cc = gd.CubicalComplex(dimensions=img.shape, top_dimensional_cells=img.flatten())
    diag = cc.persistence()
    return diag


def features_from_image(img):
    """
    Extrait des features simples depuis le diagramme de persistance d'une image.
    """
    diag = persistence_from_image(img)
    points = np.array([pt for _, pt in diag if np.isfinite(pt[1])])
    if points.size == 0:
        return np.zeros(8)

    births, deaths = points[:,0], points[:,1]
    pers = deaths - births
    pmax = pers.max() if len(pers) else 0.0
    eps = 1e-3 if pmax == 0 else 0.01 * pmax  # seuil pour séparer points persistants vs quasi-nuls
    n_points = len(points)
    n_diag = np.sum(pers < eps)
    n_persistent = np.sum(pers >= eps)
    total_persistence = pers.sum()
    max_persistence = pers.max()
    mean_persistence = pers.mean()
    mean_birth = births.mean()
    mean_death = deaths.mean()
    return np.array([
        n_points, n_diag, n_persistent,
        total_persistence, max_persistence,
        mean_persistence, mean_birth, mean_death
    ])


def tda_features_from_images(images, n_levels=5, mode="quantile"):
    """
    Pipeline complet pour extraire les features TDA à partir des images de persistance.
    Combine les features de plusieurs niveaux de superlevel filtration.
    """
    all_features = []
    for img in images:
        filtrations = superlevel_filtration(img, n_levels=n_levels, mode=mode)
        feats_levels = [features_from_image(fimg) for _, fimg in filtrations]
        feats_concat = np.concatenate(feats_levels)  # concatène les features de tous les niveaux
        all_features.append(feats_concat)
    return np.array(all_features)


def compute_final_tda_features_filtered_only(adata, images, n_levels=5):
    """
    Extrait uniquement les features TDA issues des images filtrées (robustes).
    Résultat stocké dans adata.obsm["tda_features_final"].
    """
    # Features à partir des images de persistance
    tda_feat_img = tda_features_from_images(images, n_levels=n_levels, mode="quantile")  # (n_cells, n_levels*8)

    # Normalisation
    scaler = StandardScaler()
    tda_feat_img_scaled = scaler.fit_transform(tda_feat_img)

    # Stockage dans AnnData
    adata.obsm["tda_features_final"] = tda_feat_img_scaled
    return adata
