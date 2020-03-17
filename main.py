import numpy as np
import csv
import networkx as nx
import matplotlib.pyplot as plt
import HCS
import time
import networkx.algorithms.community as community
import sklearn.cluster as cluster
from scipy.cluster.hierarchy import dendrogram
import os
import json
import sklearn

NCARDS = 1711
DATA_PATH = "C:/Users/joele/OneDrive/Games/Real Hearthstone planning/Data"
INDEX_REF = dict()
CONN_MATRIX_NEUTRALS = np.zeros((1000, 1000))

# Find the highest-scored cards using an iterative process. Pick the top card, then remove all connections from the
# network connecting to that card. Repeat.
def find_top_card_neighbourhood_deletion(print_counts, bundle_dict):
    print_counts = {k: v for k, v in sorted(print_counts.items(), key=lambda card: card[1], reverse=True)}
    while (list(print_counts.values())[0] > 0):
        # Find the cards with the highest scores
        scores = list(print_counts.values())
        card1_score = scores[0]
        i = 1
        card2_score = scores[1]
        while card2_score == card1_score:
            i += 1
            card2_score = scores[i]
        n_top_cards = i
        curr_top_cards = list(print_counts.keys())[0:i]
        for curr_card_name in curr_top_cards:
            # Find all the bundles that card belongs to
            for bundle_name in bundle_dict.keys():
                if curr_card_name in bundle_dict[bundle_name]:
                    # For all cards in those bundles, give them -1 point
                    for related_card_name in bundle_dict[bundle_name]:
                        print_counts[related_card_name] -= 1
            print_counts = {k: v for k, v in sorted(print_counts.items(), key=lambda card: card[1], reverse=True)}


# Get all the cards with print_count of >= 6. Then add cards outside this group with the highest synergy
# with the group, and remove cards with lowest synergy within the group.
# Repeat this until the lowest in-group synergy is equal or higher than the highest outof-group synergy
def polish_best_cards_group():
    included_card_synergy_ratings = {}
    excluded_card_synergy_ratings = {}
    for card_name in print_counts.keys():
        if print_counts[card_name] >= 6:
            included_card_synergy_ratings[card_name] = 0
        else:
            excluded_card_synergy_ratings[card_name] = 0
    for card_name in included_card_synergy_ratings.keys():
        card_synergy = 0
        for other_card in included_card_synergy_ratings.keys():
            card_synergy += CONN_MATRIX[all_cards_list.index(card_name)][all_cards_list.index(other_card)]
        included_card_synergy_ratings[card_name] = card_synergy
    for card_name in excluded_card_synergy_ratings.keys():
        card_synergy = 0
        for other_card in excluded_card_synergy_ratings.keys():
            card_synergy += CONN_MATRIX[all_cards_list.index(card_name)][all_cards_list.index(other_card)]
        excluded_card_synergy_ratings[card_name] = card_synergy

    added_score = 1
    removed_score = 0
    i = 1
    while True:
        included_card_synergy_ratings = {k: v for k, v in
                                         sorted(included_card_synergy_ratings.items(),
                                                key=lambda card_name: card_name[1],
                                                reverse=False)}
        excluded_card_synergy_ratings = {k: v for k, v in
                                         sorted(excluded_card_synergy_ratings.items(),
                                                key=lambda card_name: card_name[1],
                                                reverse=True)}
        card_to_remove = list(included_card_synergy_ratings.keys())[0]
        card_to_add = list(excluded_card_synergy_ratings.keys())[0]
        removed_score = included_card_synergy_ratings[card_to_remove]
        added_score = excluded_card_synergy_ratings[card_to_add]
        if added_score <= removed_score:
            break
        print(str(i))
        print("Removing " + card_to_remove + " with score " + str(removed_score))
        print("Adding " + card_to_add + " with score " + str(added_score))
        excluded_card_synergy_ratings[card_to_remove] = included_card_synergy_ratings[card_to_remove]
        included_card_synergy_ratings.pop(card_to_remove)
        included_card_synergy_ratings[card_to_add] = excluded_card_synergy_ratings[card_to_add]
        excluded_card_synergy_ratings.pop(card_to_add)
        # Update the synergy ratings to account for the card removed and one added
        # Get all the neighbours of the card
        card_to_remove_connections = all_cards_neighbours[card_to_remove]
        for card_neighbour in card_to_remove_connections:
            # Subtract the card to remove's connection strengths from its neighbours' synergy scores
            if card_neighbour in included_card_synergy_ratings:
                included_card_synergy_ratings[card_neighbour] -= CONN_MATRIX[all_cards_list.index(card_to_remove)][
                    all_cards_list.index(card_neighbour)]
            else:
                excluded_card_synergy_ratings[card_neighbour] -= CONN_MATRIX[all_cards_list.index(card_to_remove)][
                    all_cards_list.index(card_neighbour)]
        card_to_add_connections = all_cards_neighbours[card_to_add]
        for card_neighbour in card_to_add_connections:
            # Add the card_to_add's connections strengths from its neighbours' synergy scores
            if card_neighbour in included_card_synergy_ratings:
                included_card_synergy_ratings[card_neighbour] += CONN_MATRIX[all_cards_list.index(card_to_remove)][
                    all_cards_list.index(card_neighbour)]
            else:
                excluded_card_synergy_ratings[card_neighbour] += CONN_MATRIX[all_cards_list.index(card_to_remove)][
                    all_cards_list.index(card_neighbour)]
        i += 1

# Calculate the scores between each bundle of cards, using shared cards
def bundle_dist_scores(bundle_dict):
    bundle_dist_mat = np.zeros((len(bundle_dict), len(bundle_dict)))
    for bundle_a in bundle_dict:
        for bundle_b in bundle_dict:
            overlap_count = 0
            for card in bundle_dict[bundle_a]:
                if card in bundle_dict[bundle_b]:
                    overlap_count += 1
            # Calculate the F1 score
            if overlap_count == 0:
                F1 = 0
            else:
                prec = overlap_count / len(bundle_dict[bundle_a])
                rec = overlap_count / len(bundle_dict[bundle_b])
                F1 = 2 / (1 / prec + 1 / rec)
            # Save the score
            bundle_dist_mat[list(bundle_dict.keys()).index(bundle_a), list(bundle_dict.keys()).index(bundle_b)] = F1
    return bundle_dist_mat

# Find the closest neighbours for a group of cards
def closest_neighbours_for_card_group(card_connections, card_group):
    group_neighbours_list = {}
    for card_name in card_group:
        card_neighbour_list = card_connections[card_name]
        for card_neighbour in card_neighbour_list:
            if card_neighbour in group_neighbours_list:
                group_neighbours_list[card_neighbour] += card_connections[card_name][card_neighbour]
            else:
                group_neighbours_list[card_neighbour] = card_connections[card_name][card_neighbour]
    group_neighbours_list = {k: v for k, v in sorted(group_neighbours_list.items(), key=lambda card: card[1], reverse=True)}
    return group_neighbours_list

# Normalize the strengths of connections by the classes of cards that are at the ends of the connections
def normalize_card_connections_by_class(card_connections, card_data):
    list_neutral_neutral = []
    list_neutral_class = []
    list_class_class_same = []
    list_class_class_different = []
    for card_a in card_connections:
        card_neighbours = card_connections[card_a]
        for card_b in card_neighbours:
            if card_a != card_b:
                card_a_class = card_data[card_a]['cardClass']
                card_b_class = card_data[card_b]['cardClass']
                conn_stren = card_connections[card_a][card_b]
                if card_a_class == 'NEUTRAL' or card_b_class == 'NEUTRAL':
                    if card_a_class == 'NEUTRAL' and card_b_class == 'NEUTRAL':
                        list_neutral_neutral.append(conn_stren)
                    else:
                        list_neutral_class.append(conn_stren)
                else:
                    if card_a_class == card_b_class:
                        list_class_class_same.append(conn_stren)
                    else:
                        list_class_class_different.append(conn_stren)
    avg_neutral_neutral = np.sum(list_neutral_neutral)/len(list_neutral_neutral)
    avg_neutral_class = np.sum(list_neutral_class)/len(list_neutral_class)
    avg_class_class_same = np.sum(list_class_class_same)/len(list_class_class_same)
    avg_class_class_different = np.sum(list_class_class_different)/len(list_class_class_different)

    card_connections_normed = {}
    for card_a in card_connections:
        card_connections_normed[card_a] = {}
        card_neighbours = card_connections[card_a]
        for card_b in card_neighbours:
            if card_a != card_b:
                card_a_class = card_data[card_a]['cardClass']
                card_b_class = card_data[card_b]['cardClass']
                conn_stren = card_connections[card_a][card_b]
                if card_a_class == 'NEUTRAL' or card_b_class == 'NEUTRAL':
                    if card_a_class == 'NEUTRAL' and card_b_class == 'NEUTRAL':
                        card_connections_normed[card_a][card_b] = conn_stren/avg_neutral_neutral
                    else:
                        card_connections_normed[card_a][card_b] = conn_stren/avg_neutral_class
                else:
                    if card_a_class == card_b_class:
                        card_connections_normed[card_a][card_b] = conn_stren/avg_class_class_same
                    else:
                        card_connections_normed[card_a][card_b] = conn_stren/avg_class_class_different
    for (card_name, card_neighbours) in card_connections_normed.items():
        card_neighbours = {k:v for k,v in sorted(card_neighbours.items(), key=lambda card: card[1], reverse=True)}
        card_connections_normed[card_name] = card_neighbours
    return card_connections_normed

# Find out what percentage of each bundle's cards I have already printed
def find_bundles_with_least_prints_required(own_printed_cards, bundle_dict):
    bundle_completion_rates = {}
    top_bundles = []
    own_printed_cards_copy = own_printed_cards
    for i in range(0, 15):
        for bundle_name in bundle_dict:
            bundle_size = len(bundle_dict[bundle_name])
            k = 0
            for card_name in bundle_dict[bundle_name]:
                if card_name in own_printed_cards_copy:
                    k += 1

            bundle_completion_rates[bundle_name] = k / bundle_size
        bundle_completion_rates = {k: v for k, v in
                                   sorted(bundle_completion_rates.items(), key=lambda bundle: bundle[1], reverse=True)}
        top_bundle = list(bundle_completion_rates.keys())[0]
        top_bundles.append(top_bundle)
        for card_name in bundle_dict[top_bundle]:
            if card_name in own_printed_cards_copy:
                own_printed_cards_copy.remove(card_name)
    return bundle_completion_rates

# Get a dict where each key is a card name and each value is a dict with
# the names and connection strenghts of neighbouring cards
def get_cards_neighbours(cards_list, bundle_dict):
    cards_neighbours = {}
    for card_name in cards_list:
        card_neighbours = {}
        for bundle_name in bundle_dict.keys():
            bundle_cards = bundle_dict[bundle_name]
            if card_name in bundle_cards:
                for neighbour_card in bundle_cards:
                    if neighbour_card in card_neighbours:
                        # Normalize connection strength by neighbour's print count
                        #card_neighbours[neighbour_card] += 1 / print_counts[neighbour_card]
                        card_neighbours[neighbour_card] += 1
                    else:
                        # Normalize connection strength by neighbour's print count
                        #card_neighbours[neighbour_card] = 1 / print_counts[neighbour_card]
                        card_neighbours[neighbour_card] = 1
        card_neighbours = {k: v for k, v in sorted(card_neighbours.items(), key=lambda card: card[1], reverse=True)}
        cards_neighbours[card_name] = card_neighbours
    # Sort card neighbourhoods by card print counts
    cards_neighbours = {k: v for k, v in
                            sorted(cards_neighbours.items(), key=lambda card_name: print_counts[card_name[0]],
                                   reverse=True)}
    return cards_neighbours

# Get the average strength of a card's connections, for each card
def get_average_card_conn_strength(cards_neighbours, cards_list):
    avg_card_conn_strengths = {}
    for card_name in cards_neighbours:
        if card_name in cards_list:
            neighbour_counts = list(normalized_card_conns[card_name].values())
            # Drop the card's connection to itself, which is just its print count
            # neighbour_counts = neighbour_counts[1:]
            card_score = np.sum(neighbour_counts)/len(neighbour_counts)
            avg_card_conn_strengths[card_name] = card_score
    avg_card_conn_strengths = {k:v for k,v in sorted(avg_card_conn_strengths.items(), key=lambda card_name: avg_card_conn_strengths[card_name[0]], reverse=True)}
    return avg_card_conn_strengths

# Calculate the connection strengths between each pair of cards and store them in a matrix
def get_card_conn_matrix(cards_list, score_type):
    conn_matrix = np.zeros((NCARDS, NCARDS))
    for i in range(0, len(cards_list)):
        for j in range(0, len(cards_list)):
            card1 = cards_list[i]; card2 = cards_list[j]
            if card2 in all_cards_neighbours[card1]:
                score1 = all_cards_neighbours[card1][card2]
                # Use F1 score
                if score_type == "F1":
                    score1 = score1/print_counts[card2]
            else:
                score1 = 0
            if card1 in all_cards_neighbours[card2]:
                score2 = all_cards_neighbours[card2][card1]
                # Use F1 score
                if score_type == "F1":
                    score2 = score2/print_counts[card1]
            else:
                score2 = 0
            if score1 == 0 or score2 == 0:
                score = 0
            else:
                # Harmonic mean
                score = 2*score1*score2/(score1+score2)

            conn_matrix[i][j] = score
            conn_matrix[j][i] = score
    return conn_matrix

# Make the network of cards "sharper" by applying a filter to it, multiple times. In theory this should accumulate mass
# (scores) to existing dense areas, thus highlighting cards with both high scores and highly-scoring neighbourhoods.
def sharpen_network(print_counts, cards_neighbours):
    for k in range(0, 10):
        prev_scores = print_counts.copy()
        new_scores = print_counts.copy()
        for card_name in all_cards_neighbours:
            new_value_card = 0
            n_neighbours = len(all_cards_neighbours[card_name])
            for card_neighbour in all_cards_neighbours[card_name]:
                # Don't count the card itself
                if card_neighbour != card_name:
                    new_value_card -= prev_scores[card_neighbour]
            new_value_card += (n_neighbours+1)*prev_scores[card_name]
            new_value_card = np.minimum(np.maximum(new_value_card, 0), 1000000000)
            new_scores[card_name] = new_value_card
        prev_scores = new_scores
        n_nonzero_elements = len(np.nonzero(list(new_scores.values()))[0])
    return new_scores

# Find different kinds of local maxima of scores in a network
def find_network_local_maxima(print_counts, cards_neighbours):
    print_count_local_maxima = []
    neighbour_count_local_maxima = []
    for card_name in cards_neighbours:
        neighbour_count = len(cards_neighbours[card_name])
        card_print_count = print_counts[card_name]
        neighbour_maxima_found = 1
        print_maxima_found = 1
        for neighbour_name in cards_neighbours[card_name]:
            if len(cards_neighbours[neighbour_name]) > neighbour_count:
                neighbour_maxima_found = 0
            if print_counts[neighbour_name] > card_print_count:
                print_maxima_found = 0
        if neighbour_maxima_found:
            neighbour_count_local_maxima.append(card_name)
        if print_maxima_found:
            print_count_local_maxima.append(card_name)

# Calculate scores for bundles using cards' print counts
def get_bundle_scores_by_print_count(print_counts, bundle_dict):
    bundle_scores = {}
    for bundle_name in bundle_dict:
        bundle_cards = bundle_dict[bundle_name]
        bundle_score = 0
        for card_name in bundle_cards:
            bundle_score += print_counts[card_name]
        bundle_score /= len(bundle_dict[bundle_name])
        bundle_scores[bundle_name] = bundle_score
    bundle_scores = {k: v for k, v in sorted(bundle_scores.items(), key=lambda bundle: bundle[1], reverse=True)}
    return bundle_scores

def fit_kmeans_models(conn_matrix, clusters_min, clusters_max):
    inertias = []
    for k in range(clusters_min, clusters_max):
        # Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
        kmeans_model = cluster.KMeans(n_clusters=k, random_state=1).fit(conn_matrix)

        # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
        labels = kmeans_model.labels_

        # Sum of distances of samples to their closest cluster center
        inertia = kmeans_model.inertia_
        inertias.append(inertia)
        print("k:", k, " cost:", inertia)

    plt.figure()
    plt.plot(inertias)

# Make a histogram of the distances between bundles
def get_bundles_distance_histogram(bundle_dist_mat):
    bundle_dists_list = []
    # Turn matrix to vector
    bundle_dists_vec = np.reshape(bundle_dist_mat, (len(bundle_dist_mat) ** 2,))
    bundle_dists_vec = bundle_dists_vec[np.nonzero(bundle_dists_vec)]
    bundle_dists_vec = np.sort(bundle_dists_vec)
    bundle_dist_hist = np.histogram(bundle_dists_vec, bins=100)
    plt.figure()
    plt.hist(bundle_dists_vec, bins=100)

# Draw a network of bundles with nodes and connections and save it
def plot_connections_network(bundle_dist_mat, bundle_dict):
    bundlename_ref = {k: v for k, v in zip(range(0, len(bundle_dict.keys())), bundle_dict.keys())}
    G_01 = nx.from_numpy_matrix(bundle_dist_mat)
    plt.figure()
    node_poses = nx.spring_layout(G_01)
    nx.draw_networkx_nodes(G_01, node_poses, node_size = 8, labels=bundlename_ref)
    nx.draw_networkx_edges(G_01, node_poses, width=0.5)
    nx.draw_networkx_labels(G_01, node_poses, node_size = 8, labels=bundlename_ref, font_size=2)
    plt.savefig("Networkplot.png", dpi=700)

#This matrix only keeps the connections that mark nearest neighbor relations, and makes them all equal
def get_nearest_neighbour_matrix(conn_matrix):
    nearest_neighbor_matrix = np.zeros((NCARDS, NCARDS))
    for i in range(0, NCARDS):
        card_strongest_conn = 0
        for j in range(0, NCARDS):
            # Find the strongest connection
            if conn_matrix[i][j] > card_strongest_conn:
                card_strongest_conn = conn_matrix[i][j]
        if card_strongest_conn == 0:
            hallo = 5
        for j in range(0, NCARDS):
            if conn_matrix[i][j] == card_strongest_conn:
                nearest_neighbor_matrix[i][j] = 1
    return nearest_neighbor_matrix

print_counts = {}
with open('C:\\Users\\joele\\OneDrive\\Games\\Real Hearthstone planning\\Data\\Dungeon Run bundles no Legends.csv') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';', quotechar='"')
    bundle_names = list()
    bundle_dict = {}
    next_dict_index = 0
    for row in csvreader:
        # Filter out all the empty strings
        row = list(filter(None, row))
        if len(row) != 0:
            # Put all the bundles in a dict
            bundle_name = row[0]
            i = 0
            # Some bundles have same names, so add next free index at the end
            while bundle_name in bundle_dict:
                bundle_name = row[0] + str(i+2)
                i += 1
            bundle_dict[bundle_name] = row[1:]
            # Update the card print counts
            for card_name in row[1:]:
                if card_name in print_counts.keys():
                    print_counts[card_name] += 1
                else:
                    print_counts[card_name] = 1
all_cards_list = list(print_counts.keys())
# Sort alphabetically
print_counts = {k: v for k, v in sorted(print_counts.items(), key=lambda card: card[1], reverse=True)}

json_file = open(DATA_PATH + "/" + "cards.collectible.json", encoding='utf-8')
json_str = json_file.read()
cards_list_coll = json.loads(json_str)
all_card_data = {}
for card_info in cards_list_coll:
    if card_info['set'] != 'HERO_SKINS':
        all_card_data[card_info['name']] = card_info
cards_list_neutral = []

# Make a card list with only neutral cards
for i in range(0, len(cards_list_coll)):
    card_data = cards_list_coll[i]
    if card_data['name'] in all_cards_list and card_data['cardClass'] == "NEUTRAL":
        cards_list_neutral.append(card_data['name'])

# Make a copy of bundle_dict with only neutral cards in bundles
bundle_dict_neutral = {}
for bundle_name in bundle_dict:
    bundle_dict_neutral[bundle_name] = []
    for card_name in bundle_dict[bundle_name]:
        if card_name in cards_list_neutral:
            bundle_dict_neutral[bundle_name].append(card_name)

with open('C:\\Users\\joele\\OneDrive\\Games\\Real Hearthstone planning\\Data\\Own printed cards.csv') as csvfile:
    own_printed_cards = []
    csvreader = csv.reader(csvfile, delimiter=';', quotechar='"')
    for row in csvreader:
        own_printed_cards.append(row[0])

find_top_card_neighbourhood_deletion(print_counts, bundle_dict)

all_cards_neighbours = get_cards_neighbours(all_cards_list, bundle_dict)

normalized_card_conns = normalize_card_connections_by_class(all_cards_neighbours, all_card_data)

avg_card_dists_neutral = get_average_card_conn_strength(all_cards_neighbours, cards_list_neutral)

conn_matrix = get_card_conn_matrix(all_cards_list, score_type='counts')
bundle_scores = get_bundle_scores_by_print_count(print_counts, bundle_dict)

bundle_dist_mat = bundle_dist_scores(bundle_dict)

for i in range(0, len(row)):
    if i == 0:
        # Get the name of the bundle
        bundle_names.append(row[i])
    elif row[i] != "":
        card_name_a = row[i]
        if card_name_a not in INDEX_REF:
            INDEX_REF[card_name_a] = next_dict_index
            dict_index_a = next_dict_index
            next_dict_index += 1
        else:
            dict_index_a = INDEX_REF[card_name_a]
        for j in range(i+1, len(row)):
            if row[j] != "":
                card_name_b = row[j]
                if card_name_b not in INDEX_REF:
                    INDEX_REF[card_name_b] = next_dict_index
                    dict_index_b = next_dict_index
                    next_dict_index += 1
                else:
                    dict_index_b = INDEX_REF[card_name_b]

                conn_matrix[dict_index_a][dict_index_b] += 1
                conn_matrix[dict_index_b][dict_index_a] += 1

# dict with indices as keys and names as values
cardname_ref = {v: k for k, v in list(INDEX_REF.items())[0:NCARDS]}
bundlename_ref = {k: v for k, v in zip(range(0, len(bundle_dict.keys())), bundle_dict.keys())}

nearest_neighbour_matrix = get_nearest_neighbour_matrix(conn_matrix)

G = nx.from_numpy_matrix(conn_matrix)

G_nearest = nx.from_numpy_matrix(nearest_neighbour_matrix)

conn_comps = [G_nearest.subgraph(c).copy() for c in nx.connected_components(G_nearest)]

# Perform hierarchical clustering, with different amounts of clusters
for c in range(20, 50):
    bundles_hier_clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=c, connectivity=bundle_dist_mat).fit(bundle_dist_mat)
    labels_dict = {}
    for i in range(0, len(bundles_hier_clusters.labels_)):
        label = bundles_hier_clusters.labels_[i]
        if label in labels_dict.keys():
            labels_dict[label].append(i)
        else:
            labels_dict[label] = [i]

    # Go through each cluster
    for cluster in labels_dict.keys():
        # Go through each bundle in given cluster
        for bundle_index in labels_dict[cluster]:
            # Print bundle names by cluster
            bundle_name = list(bundle_dict.keys())[bundle_index]
            print(bundle_name)
        print("\n")

# Partition the graph into highly connected components
#G_partitioned = HCS.HCS(conn_comps[0])
# Save the graph
#nx.write_gml(G_partitioned, 'C:\\Users\\joele\\OneDrive\\Games\\Real Hearthstone planning\\Data\\G_partitioned.gml')
# Load the graph
G_parted = nx.read_gml('C:\\Users\\joele\\OneDrive\\Games\\Real Hearthstone planning\\Data\\G_partitioned.gml')
# Remove the nodes not in the biggest connected component
bundlename_ref_parted = bundlename_ref
bundlename_ref_parted.pop(35); bundlename_ref_parted.pop(59); bundlename_ref_parted.pop(95)
# Convert the int keys to strings
bundlename_ref_parted_keys_str = list(map(str, list(bundlename_ref_parted.keys())))
bundlename_ref_parted_str = dict(list(zip(bundlename_ref_parted_keys_str, list(bundlename_ref_parted.values()))))


# Plot the partitioned graph
plt.figure()
node_poses = nx.spring_layout(G_parted)
nx.draw_networkx_nodes(G_parted, node_poses, node_size = 50, labels=bundlename_ref_parted_str)
nx.draw_networkx_labels(G_parted, node_poses, node_size = 50, labels=bundlename_ref_parted_str)
#nx.draw_networkx_edges(G_partitioned, node_poses)

conn_comps_G_parted = [G_parted.subgraph(c).copy() for c in nx.connected_components(G_parted)]

start = time.time()
c = list(community.greedy_modularity_communities(G))
end = time.time()
total_time = end - start
G_coverage = community.coverage(G, c)

plt.figure()
for i in range(0, len(c)):
    hallo = G_nearest.subgraph(c[i])
    node_poses_nearest = nx.spring_layout(G_nearest.subgraph(c[i]))
    nx.draw_networkx_nodes(G_nearest.subgraph(c[i]), node_poses_nearest, node_size=50, labels=cardname_ref)
    nx.draw_networkx_labels(G_nearest.subgraph(c[i]), node_poses_nearest, node_size=50, labels=cardname_ref)
    plt.show()

plt.figure()
plt.imshow(conn_matrix)
plt.show()

plt.figure()
plt.imshow(nearest_neighbour_matrix)
plt.show()