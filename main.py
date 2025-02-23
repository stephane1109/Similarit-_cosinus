# pip install spacy networkx matplotlib pyvis scipy
# python -m spacy download fr_core_news_lg

import spacy
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cosine
from pyvis.network import Network
from collections import Counter
import itertools
from networkx.algorithms.community import greedy_modularity_communities

# Charger le modèle SpaCy avec les vecteurs sémantiques
nlp = spacy.load("fr_core_news_lg")

# Charger le texte depuis le fichier et le normaliser
with open("/Users/psychiatrie-darmanin-clean.txt", "r", encoding="utf-8") as f:
    texte = f.read().lower().strip()  # Passage en minuscule et suppression des espaces inutiles

doc = nlp(texte)

# Découper le texte en articles selon "****"
articles = texte.split("****")

# Nettoyage : supprimer la première ligne de chaque article
clean_articles = []
for article in articles:
    lignes = article.strip().split("\n")  # Découper en lignes
    if len(lignes) > 1:  # Vérifier qu'il y a bien du contenu après la ligne de métadonnées
        clean_articles.append("\n".join(lignes[1:]))  # Supprimer uniquement la première ligne

# Vérification : afficher les premiers articles après nettoyage
print("DEBUG: Vérification des articles après suppression des métadonnées")
for i, article in enumerate(clean_articles[:3]):  # Afficher les 3 premiers articles nettoyés
    print(f"Article {i+1} (longueur : {len(article.split())} mots) :\n{article[:300]}...")

######## Dans la persective future de réaliser TF*IDF en amont
# Traiter chaque article séparément avec SpaCy
########
docs = [nlp(article) for article in clean_articles if article.strip()]

# Extraction des termes importants (NOMS et NOMS PROPRES uniquement, sans stopwords et sans lemmatisation)
extracted_terms = []
pos_mapping = {}

for i, doc in enumerate(docs):  # Parcours des articles analysés
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and token.is_alpha:
            original_term = token.text  # Utilisation de la forme originale
            extracted_terms.append(original_term)
            pos_mapping[original_term] = token.pos_  # Associer le mot à son POS

    # Débogage : Afficher les 10 premiers termes extraits de l'article en cours
    print(f"DEBUG: Termes extraits de l'article {i+1} ({len(extracted_terms)} termes) :")
    print(extracted_terms[:10])  # Afficher seulement les 10 premiers termes

# Vérification finale des POS des termes extraits
print("DEBUG: Vérification des POS des termes extraits (sans lemmatisation)")
for term in extracted_terms[:10]:  # Afficher 10 échantillons
    print(f"{term} → {pos_mapping.get(term, 'N/A')}")

# Compter les fréquences des termes
term_frequencies = Counter(extracted_terms)

# Garder uniquement les 50 termes les plus fréquents
most_common_terms = {term for term, _ in term_frequencies.most_common(50)}

# Filtrer les termes qui ont des vecteurs SpaCy valides
valid_terms = {term: nlp(term).vector for term in most_common_terms if nlp(term).vector_norm > 0}

# Seuil de similarité ajusté pour connecter plus de nœuds
similarity_threshold = 0.4 # Pour plus de connexions 0.3 et moins de connexion 0.6

# Création du graphe basé uniquement sur la similarité sémantique
graph = nx.Graph()

# Ajouter les liens entre termes similaires
for term1, term2 in itertools.combinations(valid_terms.keys(), 2):
    vec1, vec2 = valid_terms[term1], valid_terms[term2]
    similarity = 1 - cosine(vec1, vec2)
    if similarity >= similarity_threshold:
        graph.add_edge(term1, term2, weight=float(similarity))  # Convertir en float standard

# Supprimer les nœuds sans connexions (isolés)
isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]
graph.remove_nodes_from(isolated_nodes)

# Détection des communautés (groupes de mots sémantiquement proches)
communities = list(greedy_modularity_communities(graph))

# Attribution des couleurs aux communautés
def generate_color_palette(n):
    colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A8", "#A833FF", "#33FFF2", "#FF8C00", "#FFD700"]
    return [colors[i % len(colors)] for i in range(n)]

community_colors = generate_color_palette(len(communities))
color_map = {}
for i, community in enumerate(communities):
    for node in community:
        color_map[node] = community_colors[i]

# Création du graphe interactif avec Pyvis
net = Network(notebook=False, height="800px", width="100%", bgcolor="#222222", font_color="white", directed=False)

# Ajouter les noeuds à Pyvis avec couleur et communauté
for node in graph.nodes():
    pos_tag = pos_mapping.get(node, "UNK")  # Récupérer le POS ou "UNK" (inconnu)
    net.add_node(
        node,
        size=graph.degree(node) * 5,
        title=f"{node} ({pos_tag}): {term_frequencies.get(node, 0)} occurrences",
        label=f"{node} ({pos_tag})",  # Affichage du POS à côté du mot
        color=color_map.get(node, "#FFFFFF"),
        borderWidth=4,
        shadow=True
    )

# Ajouter les arêtes colorées selon la force de la relation
for edge in graph.edges(data=True):
    weight = edge[2]['weight']
    net.add_edge(
        edge[0], edge[1],
        value=weight * 5,  # Amplification pour meilleure visualisation
        color=f"rgba(255, 255, 255, {weight})",  # Transparence selon le poids
        title=f"Score: {weight:.2f}"  # Ajout du score de similarité sur l'arête
    )

# Activer l'interface interactive
net.show_buttons(filter_=['physics'])

# Générer et sauvegarder le graphe interactif
net.write_html("graph_similarité_cosinus.html")

print("Graphe de similarité cosinus généré : ouvrez 'graph_similarité_cosinus.html' dans votre navigateur.")

# Liste pour stocker les relations entre les mots et leur score de similarité
similarity_data = []

# Remplissage de la liste avec les paires de mots et leur score de similarité
for edge in graph.edges(data=True):
    term1, term2, weight = edge[0], edge[1], edge[2]['weight']
    similarity_data.append([term1, term2, round(weight, 3)])  # Arrondi à 3 décimales

# Création d'un DataFrame Pandas
df_similarity = pd.DataFrame(similarity_data, columns=["Terme 1", "Terme 2", "Score Similarité"])

# Sauvegarde du fichier CSV
csv_path = "similarite_termes.csv"
df_similarity.to_csv(csv_path, index=False, encoding="utf-8")

print(f"Fichier CSV avec scores de similarité enregistré sous : {csv_path}")
