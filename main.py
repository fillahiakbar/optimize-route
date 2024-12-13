import tensorflow as tf
import numpy as np
import random
from collections import deque
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic
from scipy.spatial import cKDTree
import logging
from google.cloud import firestore  # Firestore library

# Logging setup
logging.basicConfig(level=logging.INFO)

# Fetch locations from Firestore
def fetch_locations_from_firestore():
    """
    Fetch location data from Firestore.
    Returns:
        list: List of all locations [(lat, lng), ...].
        tuple: Depot location (lat, lng).
    """
    try:
        db = firestore.Client()  # Firestore client
        doc_ref = db.collection('your_collection_name').document('your_document_name')
        doc = doc_ref.get()

        if not doc.exists:
            raise ValueError("Document does not exist in Firestore!")

        data = doc.to_dict()
        if 'locations' not in data or not isinstance(data['locations'], list):
            raise ValueError("Invalid Firestore data: 'locations' field is missing or not a list.")

        locations = []
        depot_location = None
        for idx, loc in enumerate(data['locations']):
            if 'lat' in loc and 'lng' in loc:
                lat, lng = loc['lat'], loc['lng']
                if idx == 0:  # Assume the first location is always the depot
                    depot_location = (lat, lng)
                else:
                    locations.append((lat, lng))
            else:
                logging.warning(f"Invalid location entry at index {idx}: {loc}")
        
        if depot_location is None:
            raise ValueError("Depot location is missing in Firestore data.")

        return [depot_location] + locations, depot_location
    except Exception as e:
        raise RuntimeError(f"Error fetching Firestore data: {e}")

# Fetch road network
def fetch_road_network(depot_location, dist=20000, network_type='drive'):
    logging.info("Fetching road network...")
    return ox.graph_from_point(depot_location, dist=dist, network_type=network_type)

# Map locations to nearest nodes
def map_to_nearest_nodes(locations, graph, distance_threshold=2000):
    graph_nodes = {node: (data['y'], data['x']) for node, data in graph.nodes(data=True)}
    graph_node_coords = np.array(list(graph_nodes.values()))
    graph_node_ids = np.array(list(graph_nodes.keys()))

    tree = cKDTree(graph_node_coords)
    distances, indices = tree.query(np.array(locations))
    valid_nodes = []
    for idx, (dist, nearest_idx) in enumerate(zip(distances, indices)):
        if dist > distance_threshold:
            logging.warning(f"Location {locations[idx]} is too far from the network!")
        else:
            valid_nodes.append(graph_node_ids[nearest_idx])
    return valid_nodes

# Build distance matrix
def build_distance_matrix(valid_nodes, graph):
    matrix = np.zeros((len(valid_nodes), len(valid_nodes)))
    for i, start_node in enumerate(valid_nodes):
        for j, end_node in enumerate(valid_nodes):
            if i != j:
                try:
                    matrix[i, j] = nx.shortest_path_length(graph, start_node, end_node, weight='length')
                except nx.NetworkXNoPath:
                    matrix[i, j] = float('inf')
    return matrix

# Define DQN Model
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.output_layer(x)

# VRP Agent
class VRPAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(num_actions)
        self.target_model = DQN(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.replay_buffer = deque(maxlen=10000)
        self.gamma = 0.99
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.model(np.array([state]))
            q_values = q_values.numpy()[0]
            masked_q_values = np.full_like(q_values, -np.inf)
            masked_q_values[valid_actions] = q_values[valid_actions]
            return np.argmax(masked_q_values)

    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            next_q_values = self.target_model(next_states)
            max_next_q_values = np.max(next_q_values, axis=1)

            targets = rewards + (1 - dones) * self.gamma * max_next_q_values
            q_values = tf.reduce_sum(q_values * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.reduce_mean((targets - q_values) ** 2)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train VRP Agent
def train_vrp_agent(agent, distance_matrix, valid_nodes, num_episodes=100):
    best_routes = None
    best_distance = float('inf')

    for episode in range(num_episodes):
        state = 0
        total_distance = 0
        unvisited_nodes = set(range(1, len(valid_nodes)))

        while unvisited_nodes:
            action = agent.choose_action(state, list(unvisited_nodes))
            total_distance += distance_matrix[state][action]
            unvisited_nodes.remove(action)
            state = action

        if total_distance < best_distance:
            best_distance = total_distance
            best_routes = total_distance  # Simplified for demo

        agent.train()
        logging.info(f"Episode {episode + 1}: Total Distance = {total_distance}, Epsilon = {agent.epsilon}")

    return best_routes

# Main Execution
locations, depot_location = fetch_locations_from_firestore()
G = fetch_road_network(depot_location)
valid_nodes = map_to_nearest_nodes(locations, G)
distance_matrix = build_distance_matrix(valid_nodes, G)

agent = VRPAgent(len(valid_nodes))
best_routes = train_vrp_agent(agent, distance_matrix, valid_nodes)

logging.info(f"Best Routes: {best_routes}")
