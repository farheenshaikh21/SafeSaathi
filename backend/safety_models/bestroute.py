import networkx as nx
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

class RouteSafetySystem:
    def __init__(self, model_path='safety_model.pkl'):
        self.network = {
            'nodes': {},
            'edges': []
        }
        self.model = None
        self.scaler = None
        self.graph = None
        self.feature_names = None
        self.target_name = None
        self.model_path = model_path
        self.model_trained = False
        
        self._load_existing_model()

    def _load_existing_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.feature_names = model_data['feature_names']
                    self.target_name = model_data['target_name']
                    self.model_trained = True
                print("✅ Pre-trained model loaded")
            except Exception as e:
                print(f"⚠️ Model loading failed: {e}")

    def load_data_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        nodes = {}
        edges = []

        for _, row in df.iterrows():
            source = row['Area Name']
            targets = str(row['Connected Areas']).split(",")

            if source not in nodes:
                nodes[source] = {
                    'lat': row['Latitude'],
                    'lon': row['Longitude']
                }

            for target in targets:
                target = target.strip()
                if not target or target.lower() == "nan":
                    continue

                if target not in nodes:
                    nodes[target] = {'lat': None, 'lon': None}

                edges.append({
                    'source': source,
                    'target': target,
                    'distance': row['Distance from Borivali (km)'],
                    'lighting': row.get('Lighting', 0.5),
                    'crime_rate': row.get('Crime_Rate', 0.3),
                    'population_density': row.get('Population_Density', 0.6),
                    'road_condition': row.get('Road_Condition', 0.7),
                    'emergency_distance': row.get('Emergency_Distance', 0.8),
                    'street_lights': row.get('Street_Lights', 5),
                    'cctv_coverage': row.get('CCTV_Coverage', 0.4),
                    'pedestrian_traffic': row.get('Pedestrian_Traffic', 0.5)
                })

        return nodes, edges

    def load_network_data(self, nodes, edges):
        self.network['nodes'] = nodes
        self.network['edges'] = edges
        self._build_graph()
        print(f"🌐 Network loaded: {len(nodes)} nodes, {len(edges)} edges")

    def _build_graph(self):
        self.graph = nx.Graph()

        for node, data in self.network['nodes'].items():
            self.graph.add_node(node, **data)

        for edge in self.network['edges']:
            safety = self._compute_edge_safety(edge)
            weight = edge['distance'] * (1.2 - safety)

            self.graph.add_edge(
                edge['source'],
                edge['target'],
                distance=edge['distance'],
                computed_safety=safety,
                weight=weight
            )

    def _compute_edge_safety(self, edge):
        if self.model_trained:
            features = self._extract_edge_features(edge)
            scaled = self.scaler.transform([features])
            return self.model.predict(scaled)[0]
        return 0.5  # default safety

    def _extract_edge_features(self, edge):
        features = {
            'time_of_day': 19,
            'day_of_week': 2,
            'lighting_index': edge['lighting'],
            'crime_index': edge['crime_rate'],
            'population_density': edge['population_density'],
            'road_condition': edge['road_condition'],
            'emergency_distance': edge['emergency_distance'],
            'street_light_count': edge['street_lights'],
            'cctv_coverage': edge['cctv_coverage'],
            'pedestrian_traffic': edge['pedestrian_traffic']
        }
        return [features.get(f, 0) for f in self.feature_names]

    def find_top_routes(self, start, end, top_n=3):
        paths = list(nx.all_simple_paths(self.graph, start, end))
        results = []

        for path in paths:
            distance = 0
            safety = 0

            for i in range(len(path) - 1):
                edge = self.graph[path[i]][path[i + 1]]
                distance += edge['distance']
                safety += edge['computed_safety']

            avg_safety = safety / (len(path) - 1)
            score = avg_safety / distance
            results.append((path, score))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]


# ===================== TERMINAL EXECUTION =====================
if __name__ == "__main__":
    print("\n=== ROUTE SAFETY SYSTEM ===\n")

    system = RouteSafetySystem()

    csv_path = "borivali_dahisar_safety_routes_updated.csv"
    nodes, edges = system.load_data_from_csv(csv_path)
    system.load_network_data(nodes, edges)

    start = input("Enter START location: ").strip()
    end = input("Enter END location: ").strip()

    routes = system.find_top_routes(start, end)

    print("\n🔐 Safest Routes:\n")
    for i, (path, score) in enumerate(routes, 1):
        print(f"🏆 Route {i}")
        print("   Path :", " ➔ ".join(path))
        print(f"   Score: {score:.4f}\n")
