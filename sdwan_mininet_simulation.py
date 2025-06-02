#!/usr/bin/env python3
###############################################
# File: sdwan_mininet_simulation_fixed.py
# Version corrigée avec gestion des erreurs NaN
###############################################

import os
import time
import threading
import signal
import sys
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from mininet.net import Mininet
from mininet.node import RemoteController, Host
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel
from mininet.cli import CLI
import subprocess
import random


######################################################################################
##################################### CONFIG MININET ###################################

###############################################
# Ajout à sdwan_mininet_simulation_fixed.py
# Visualisation automatique de la topologie
###############################################

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def visualize_topology(save_path=None):
    """
    Crée une visualisation professionnelle de la topologie SD-WAN
    """
    print("[INFO] Génération de la visualisation de topologie...")
    
    # Création du graphe
    G = nx.Graph()
    
    # === AJOUT DES NOEUDS ===
    
    # Hôtes Branch A
    branch_a_hosts = [f'h{i}-a' for i in range(1, HOSTS_PER_BRANCH + 1)]
    for host in branch_a_hosts:
        G.add_node(host, node_type='host', branch='A', ip=f'10.1.0.{host[1]}')
    
    # Hôtes Branch B  
    branch_b_hosts = [f'h{i}-b' for i in range(1, HOSTS_PER_BRANCH + 1)]
    for host in branch_b_hosts:
        G.add_node(host, node_type='host', branch='B', ip=f'10.2.0.{host[1]}')
    
    # Switches
    switches = {
        's1': {'type': 'cpe', 'branch': 'A', 'name': 'CPE Branch A'},
        's2': {'type': 'cpe', 'branch': 'B', 'name': 'CPE Branch B'},
        's3': {'type': 'wan', 'path': 'MPLS', 'name': 'MPLS Core'},
        's4': {'type': 'wan', 'path': 'Fiber', 'name': 'Fiber Core'},
        's5': {'type': 'wan', 'path': '4G', 'name': '4G Core'}
    }
    
    for switch, attrs in switches.items():
        G.add_node(switch, node_type='switch', **attrs)
    
    # === AJOUT DES LIENS ===
    
    # Liens hôtes vers CPE
    for host in branch_a_hosts:
        G.add_edge(host, 's1', link_type='local', bw='100M')
    
    for host in branch_b_hosts:
        G.add_edge(host, 's2', link_type='local', bw='100M')
    
    # Liens WAN
    wan_links = [
        ('s1', 's3', {'path': 'MPLS', 'bw': '20M', 'delay': '5ms', 'color': '#2E8B57'}),
        ('s3', 's2', {'path': 'MPLS', 'bw': '20M', 'delay': '5ms', 'color': '#2E8B57'}),
        ('s1', 's4', {'path': 'Fiber', 'bw': '100M', 'delay': '10ms', 'color': '#4169E1'}),
        ('s4', 's2', {'path': 'Fiber', 'bw': '100M', 'delay': '10ms', 'color': '#4169E1'}),
        ('s1', 's5', {'path': '4G', 'bw': '10M', 'delay': '50ms', 'color': '#FF6347'}),
        ('s5', 's2', {'path': '4G', 'bw': '10M', 'delay': '50ms', 'color': '#FF6347'})
    ]
    
    for src, dst, attrs in wan_links:
        G.add_edge(src, dst, link_type='wan', **attrs)
    
    # === CRÉATION DE LA VISUALISATION ===
    
    plt.figure(figsize=(16, 12))
    
    # Positions personnalisées pour une belle disposition
    pos = {
        # Branch A (gauche)
        'h1-a': (-3, 2),
        'h2-a': (-3, 1),
        'h3-a': (-3, 0),
        's1': (-1, 1),
        
        # Branch B (droite)
        'h1-b': (3, 2),
        'h2-b': (3, 1),
        'h3-b': (3, 0),
        's2': (1, 1),
        
        # WAN Core (centre)
        's3': (0, 2),    # MPLS
        's4': (0, 1),    # Fiber  
        's5': (0, 0),    # 4G
    }
    
    # === DESSIN DES NOEUDS ===
    
    # Hôtes Branch A
    nx.draw_networkx_nodes(G, pos, nodelist=branch_a_hosts, 
                          node_color='lightblue', node_size=800, 
                          node_shape='s', alpha=0.8)
    
    # Hôtes Branch B
    nx.draw_networkx_nodes(G, pos, nodelist=branch_b_hosts,
                          node_color='lightcoral', node_size=800,
                          node_shape='s', alpha=0.8)
    
    # CPE Switches
    nx.draw_networkx_nodes(G, pos, nodelist=['s1', 's2'],
                          node_color='gold', node_size=1200,
                          node_shape='h', alpha=0.9)
    
    # WAN Switches
    wan_colors = {'s3': '#2E8B57', 's4': '#4169E1', 's5': '#FF6347'}
    for switch in ['s3', 's4', 's5']:
        nx.draw_networkx_nodes(G, pos, nodelist=[switch],
                              node_color=wan_colors[switch], node_size=1000,
                              node_shape='o', alpha=0.8)
    
    # === DESSIN DES LIENS ===
    
    # Liens locaux (gris)
    local_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('link_type') == 'local']
    nx.draw_networkx_edges(G, pos, edgelist=local_edges,
                          edge_color='gray', width=2, alpha=0.6)
    
    # Liens WAN (colorés selon le type)
    for src, dst, attrs in wan_links:
        nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)],
                              edge_color=attrs['color'], width=4, alpha=0.8)
    
    # === LABELS ===
    
    # Labels des noeuds
    labels = {}
    for node in G.nodes():
        if node.startswith('h'):
            labels[node] = node
        elif node == 's1':
            labels[node] = 'CPE-A'
        elif node == 's2':
            labels[node] = 'CPE-B'
        elif node == 's3':
            labels[node] = 'MPLS'
        elif node == 's4':
            labels[node] = 'Fiber'
        elif node == 's5':
            labels[node] = '4G'
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    # === ANNOTATIONS ET LÉGENDES ===
    
    # Titre
    plt.title('Topologie SD-WAN - Équilibrage de Charge Multi-Chemins', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Zones Branch
    branch_a_rect = Rectangle((-3.5, -0.5), 1, 3, linewidth=2, 
                             edgecolor='blue', facecolor='lightblue', alpha=0.2)
    branch_b_rect = Rectangle((2.5, -0.5), 1, 3, linewidth=2,
                             edgecolor='red', facecolor='lightcoral', alpha=0.2)
    wan_rect = Rectangle((-0.5, -0.5), 1, 3, linewidth=2,
                        edgecolor='green', facecolor='lightgreen', alpha=0.2)
    
    plt.gca().add_patch(branch_a_rect)
    plt.gca().add_patch(branch_b_rect)
    plt.gca().add_patch(wan_rect)
    
    # Labels des zones
    plt.text(-3, -0.8, 'Branch A\n10.1.0.0/24', ha='center', va='top', 
             fontsize=12, fontweight='bold', color='blue')
    plt.text(3, -0.8, 'Branch B\n10.2.0.0/24', ha='center', va='top',
             fontsize=12, fontweight='bold', color='red')
    plt.text(0, -0.8, 'WAN Core\nSD-WAN Controller', ha='center', va='top',
             fontsize=12, fontweight='bold', color='green')
    
    # Légende des liens WAN
    legend_elements = [
        mpatches.Patch(color='#2E8B57', label='MPLS (20Mbps, 5ms, Poids: 3)'),
        mpatches.Patch(color='#4169E1', label='Fiber (100Mbps, 10ms, Poids: 2)'),
        mpatches.Patch(color='#FF6347', label='4G (10Mbps, 50ms, Poids: 1)'),
        mpatches.Patch(color='gray', label='Liens Locaux (100Mbps)')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', 
               bbox_to_anchor=(1, 1), fontsize=10)
    
    # Informations techniques
    info_text = """
ARCHITECTURE SD-WAN

Algorithme: Weighted Round-Robin
Contrôleur: Ryu OpenFlow 1.3
Simulation: Mininet + TCLink

Équilibrage intelligent basé sur:
• Poids des liens WAN
• Type de trafic
• Performance en temps réel
    """
    
    plt.text(-4.5, 1, info_text, fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Paramètres d'affichage
    plt.axis('off')
    plt.tight_layout()
    
    # Sauvegarde
    if save_path is None:
        save_path = os.path.join(GRAPH_DIR, f'topology_sdwan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[INFO] ✓ Topologie sauvegardée: {save_path}")
    return save_path

# Fonction à ajouter dans main() après le lancement de Mininet
def capture_topology_during_simulation(net):
    """
    Capture la topologie pendant que la simulation tourne
    """
    print("[INFO] Capture de la topologie en cours...")
    
    # Sauvegarde automatique de la topologie
    topo_path = visualize_topology()
    
    # Optionnel: capture des informations runtime
    capture_runtime_info(net, topo_path)

def capture_runtime_info(net, topo_path):
    """
    Capture des informations runtime de Mininet
    """
    runtime_info = {
        'timestamp': datetime.now().isoformat(),
        'topology_image': topo_path,
        'hosts': [],
        'switches': [],
        'links': []
    }
    
    # Informations sur les hôtes
    for host_name in [f'h{i}-a' for i in range(1, 4)] + [f'h{i}-b' for i in range(1, 4)]:
        host = net.get(host_name)
        if host:
            runtime_info['hosts'].append({
                'name': host_name,
                'ip': host.IP(),
                'mac': host.MAC(),
                'status': 'active'
            })
    
    # Sauvegarde des infos runtime
    runtime_path = os.path.join(STATS_DIR, 'topology_runtime.json')
    with open(runtime_path, 'w') as f:
        json.dump(runtime_info, f, indent=2)
    
    print(f"[INFO] ✓ Infos runtime sauvegardées: {runtime_path}")

############################################################################################
##########################################################################################

# Configuration
RESULTS_DIR = 'results'
GRAPH_DIR = os.path.join(RESULTS_DIR, 'graphs')
LOG_DIR = 'logs'
STATS_DIR = 'stats'

# Paramètres de simulation
SIM_DURATION = 60
HOSTS_PER_BRANCH = 3
NUM_BRANCHES = 2
TRAFFIC_TYPES = ['web', 'video', 'voip', 'data']

# Paramètres des liens WAN
WAN_CONFIGS = {
    'MPLS': {'bw': 20, 'delay': '5ms', 'loss': 0, 'port': 1},
    'Fiber': {'bw': 100, 'delay': '10ms', 'loss': 0.1, 'port': 2},
    '4G': {'bw': 10, 'delay': '50ms', 'loss': 2, 'port': 3}
}

# Variable globale pour contrôler l'arrêt
simulation_running = True

def signal_handler(sig, frame):
    """Gestionnaire pour arrêt propre avec Ctrl+C"""
    global simulation_running
    print("\n[INFO] Arrêt de la simulation demandé...")
    simulation_running = False

class SDWANTopo(Topo):
    """
    Topologie SD-WAN étendue :
    - Branch A: 3 hôtes (h1-a, h2-a, h3-a) connectés à s1
    - Branch B: 3 hôtes (h1-b, h2-b, h3-b) connectés à s2  
    - 3 chemins WAN parallèles : MPLS, Fiber, 4G
    """
    
    def build(self):
        # Switches principaux
        s1 = self.addSwitch('s1')  # CPE Branch A
        s2 = self.addSwitch('s2')  # CPE Branch B
        
        # Switches WAN
        s_mpls = self.addSwitch('s3')   # Chemin MPLS
        s_fiber = self.addSwitch('s4')  # Chemin Fiber
        s_4g = self.addSwitch('s5')     # Chemin 4G
        
        # Création des hôtes - Branch A
        for i in range(1, HOSTS_PER_BRANCH + 1):
            host = self.addHost(f'h{i}-a', ip=f'10.1.0.{i}/24')
            self.addLink(host, s1)
        
        # Création des hôtes - Branch B  
        for i in range(1, HOSTS_PER_BRANCH + 1):
            host = self.addHost(f'h{i}-b', ip=f'10.2.0.{i}/24')
            self.addLink(host, s2)
        
        # Liens WAN avec caractéristiques différentes
        # MPLS - haute qualité, faible latence
        self.addLink(s1, s_mpls, cls=TCLink, 
                    bw=WAN_CONFIGS['MPLS']['bw'],
                    delay=WAN_CONFIGS['MPLS']['delay'],
                    loss=WAN_CONFIGS['MPLS']['loss'])
        self.addLink(s_mpls, s2, cls=TCLink,
                    bw=WAN_CONFIGS['MPLS']['bw'],
                    delay=WAN_CONFIGS['MPLS']['delay'], 
                    loss=WAN_CONFIGS['MPLS']['loss'])
        
        # Fiber - haute bande passante
        self.addLink(s1, s_fiber, cls=TCLink,
                    bw=WAN_CONFIGS['Fiber']['bw'],
                    delay=WAN_CONFIGS['Fiber']['delay'],
                    loss=WAN_CONFIGS['Fiber']['loss'])
        self.addLink(s_fiber, s2, cls=TCLink,
                    bw=WAN_CONFIGS['Fiber']['bw'],
                    delay=WAN_CONFIGS['Fiber']['delay'],
                    loss=WAN_CONFIGS['Fiber']['loss'])
        
        # 4G - bande passante limitée, latence élevée
        self.addLink(s1, s_4g, cls=TCLink,
                    bw=WAN_CONFIGS['4G']['bw'],
                    delay=WAN_CONFIGS['4G']['delay'],
                    loss=WAN_CONFIGS['4G']['loss'])
        self.addLink(s_4g, s2, cls=TCLink,
                    bw=WAN_CONFIGS['4G']['bw'],
                    delay=WAN_CONFIGS['4G']['delay'],
                    loss=WAN_CONFIGS['4G']['loss'])

def ensure_dirs():
    """Création des dossiers nécessaires"""
    for d in [RESULTS_DIR, GRAPH_DIR, LOG_DIR, STATS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"[INFO] Dossier créé: {d}")

def launch_mininet():
    """Lance Mininet avec la topologie étendue"""
    print("[INFO] Lancement de la topologie SD-WAN...")
    topo = SDWANTopo()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6633),
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )
    net.start()
    
    # Test de connectivité
    print("[INFO] Test de connectivité...")
    result = net.pingAll()
    if result == 0:
        print("[INFO] ✓ Tous les hôtes sont connectés")
    else:
        print(f"[WARNING] {result}% de perte dans le test ping")
    
    return net

def generate_realistic_traffic(net, host_src, host_dst, traffic_type, duration=10):
    """Génère différents types de trafic réaliste"""
    
    traffic_profiles = {
        'web': {'protocol': 'tcp', 'bw': '2M', 'pattern': 'bursty'},
        'video': {'protocol': 'udp', 'bw': '8M', 'pattern': 'continuous'},
        'voip': {'protocol': 'udp', 'bw': '64K', 'pattern': 'continuous'},
        'data': {'protocol': 'tcp', 'bw': '5M', 'pattern': 'bulk'}
    }
    
    profile = traffic_profiles.get(traffic_type, traffic_profiles['data'])
    
    try:
        if profile['protocol'] == 'udp':
            # Trafic UDP pour video/voip
            host_dst.cmd(f'iperf -s -u -p 5001 > {LOG_DIR}/iperf_{host_dst.name}_{traffic_type}.log &')
            time.sleep(1)
            host_src.cmd(f'iperf -c {host_dst.IP()} -u -p 5001 -b {profile["bw"]} -t {duration} -i 2 > {LOG_DIR}/iperf_{host_src.name}_{traffic_type}.log &')
        else:
            # Trafic TCP pour web/data
            host_dst.cmd(f'iperf -s -p 5001 > {LOG_DIR}/iperf_{host_dst.name}_{traffic_type}.log &')
            time.sleep(1)
            host_src.cmd(f'iperf -c {host_dst.IP()} -p 5001 -t {duration} -i 2 > {LOG_DIR}/iperf_{host_src.name}_{traffic_type}.log &')
        
        print(f"[TRAFFIC] {traffic_type.upper()} : {host_src.name} -> {host_dst.name} ({profile['bw']}, {duration}s)")
    
    except Exception as e:
        print(f"[ERROR] Erreur génération trafic {traffic_type}: {e}")

def run_multi_traffic_simulation(net):
    """Lance plusieurs flux de trafic simultanés"""
    print(f"[INFO] Démarrage de la simulation de trafic pour {SIM_DURATION}s...")
    
    hosts_a = [net.get(f'h{i}-a') for i in range(1, HOSTS_PER_BRANCH + 1)]
    hosts_b = [net.get(f'h{i}-b') for i in range(1, HOSTS_PER_BRANCH + 1)]
    
    # Création de plusieurs flux simultanés
    traffic_scenarios = [
        ('h1-a', 'h1-b', 'video', 30),
        ('h2-a', 'h2-b', 'web', 25), 
        ('h3-a', 'h3-b', 'voip', 40),
        ('h1-b', 'h1-a', 'data', 20),
        ('h2-b', 'h3-a', 'web', 15),
        ('h3-b', 'h2-a', 'video', 35)
    ]
    
    start_time = time.time()
    active_traffics = []
    
    while simulation_running and (time.time() - start_time) < SIM_DURATION:
        current_time = time.time() - start_time
        
        # Lancer de nouveaux flux selon le scénario
        for src_name, dst_name, t_type, start_at in traffic_scenarios:
            if abs(current_time - start_at) < 1 and (src_name, dst_name) not in active_traffics:
                src_host = net.get(src_name)
                dst_host = net.get(dst_name)
                
                # Durée aléatoire pour rendre plus réaliste
                duration = random.randint(10, 30)
                
                thread = threading.Thread(
                    target=generate_realistic_traffic,
                    args=(net, src_host, dst_host, t_type, duration)
                )
                thread.daemon = True
                thread.start()
                active_traffics.append((src_name, dst_name))
                
                print(f"[INFO] Nouveau flux démarré à t={current_time:.1f}s")
        
        time.sleep(2)  # Vérification toutes les 2 secondes
    
    print("[INFO] Période de simulation terminée, attente fin des flux...")
    time.sleep(10)  # Laisser les derniers flux se terminer

def safe_parse_controller_logs():
    """Parse sécurisé des logs du contrôleur avec gestion des erreurs"""
    controller_log = os.path.join(LOG_DIR, 'sdwan_controller.log')
    
    if not os.path.isfile(controller_log):
        print(f"[WARNING] Fichier de log non trouvé: {controller_log}")
        return [], []
    
    ports = []
    timestamps = []
    
    try:
        with open(controller_log, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # On repère les lignes contenant "Path selected: port=X"
                    if 'Path selected: port=' in line or '--> Path selected: port=' in line:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            ts_str = parts[1]  # format HH:MM:SS.mmmmmm
                            
                            # Extraction du port de manière plus robuste
                            port_part = line.strip().split('port=')
                            if len(port_part) >= 2:
                                port_str = port_part[1].split()[0]  # Premier mot après port=
                                port = int(port_str)
                                
                                timestamps.append(ts_str)
                                ports.append(port)
                            
                except ValueError as e:
                    print(f"[WARNING] Erreur parsing ligne {line_num}: {e}")
                    continue
                except IndexError as e:
                    print(f"[WARNING] Format ligne invalide {line_num}: {e}")
                    continue
                    
    except Exception as e:
        print(f"[ERROR] Erreur lecture fichier log: {e}")
        return [], []
    
    print(f"[INFO] Données extraites: {len(ports)} sélections de chemin")
    return ports, timestamps

def parse_and_visualize_results():
    """Version corrigée avec gestion robuste des erreurs NaN"""
    print("[INFO] Analyse des résultats et génération des graphiques...")
    
    # Lecture sécurisée des logs
    ports, timestamps = safe_parse_controller_logs()
    
    # Vérification des données
    if not ports or len(ports) == 0:
        print("[WARNING] Aucune sélection de chemin trouvée - Génération de données de démonstration")
        create_demo_graphs()
        return
    
    print(f"[INFO] Analyse de {len(ports)} sélections de chemin")
    
    # Conversion des timestamps en secondes relatives
    times_sec = []
    if timestamps:
        try:
            fmt = '%H:%M:%S.%f'
            import datetime
            t0 = datetime.datetime.strptime(timestamps[0], fmt)
            
            for ts in timestamps:
                try:
                    t = datetime.datetime.strptime(ts, fmt)
                    delta = (t - t0).total_seconds()
                    times_sec.append(delta)
                except ValueError:
                    # Si le format ne marche pas, utiliser l'index comme temps
                    times_sec.append(len(times_sec))
                    
        except Exception as e:
            print(f"[WARNING] Erreur conversion timestamps: {e}")
            # Fallback: utiliser des index comme temps
            times_sec = list(range(len(ports)))
    
    # Assemblage des données avec vérification
    data = {
        'port': np.array(ports, dtype=int),
        'time': np.array(times_sec, dtype=float)
    }
    
    # Création des graphiques avec gestion d'erreur
    create_safe_graphs(data)

def create_safe_graphs(data):
    """Création sécurisée des graphiques avec gestion NaN"""
    try:
        plt.figure(figsize=(15, 10))
        
        # Données pour les graphiques
        unique_ports = np.unique(data['port'])
        unique_ports = unique_ports[~np.isnan(unique_ports)]  # Retirer les NaN
        
        if len(unique_ports) == 0:
            print("[WARNING] Aucun port valide trouvé")
            create_demo_graphs()
            return
        
        # Calcul des counts avec vérification
        counts = []
        for p in unique_ports:
            count = np.sum(data['port'] == p)
            counts.append(max(0, count))  # S'assurer que count >= 0
        
        # Vérification que counts n'est pas vide
        if not counts or sum(counts) == 0:
            print("[WARNING] Aucune donnée valide pour les graphiques")
            create_demo_graphs()
            return
        
        # 1) Histogramme du nombre de sélections par port
        plt.subplot(2, 3, 1)
        labels = [f'Port {int(p)}' for p in unique_ports]
        colors = ['#2E8B57', '#4169E1', '#FF6347'][:len(unique_ports)]
        
        # S'assurer qu'il n'y a pas de NaN dans counts
        counts_clean = [c if not np.isnan(c) else 0 for c in counts]
        
        wedges, texts, autotexts = plt.pie(counts_clean, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        plt.title('Répartition du Trafic par Chemin WAN', fontsize=14, fontweight='bold')
        
        # 2) Courbe temporelle des sélections de port
        plt.subplot(2, 3, 2)
        for i, port in enumerate(unique_ports):
            mask = (data['port'] == port)
            color = colors[i % len(colors)]
            
            # Filtrer les NaN
            time_data = data['time'][mask]
            port_data = data['port'][mask]
            
            # Retirer les NaN
            valid_mask = ~(np.isnan(time_data) | np.isnan(port_data))
            time_clean = time_data[valid_mask]
            port_clean = port_data[valid_mask]
            
            if len(time_clean) > 0:
                plt.plot(time_clean, port_clean, 'o', label=f'Port {int(port)}', 
                        color=color, alpha=0.6, markersize=4)
        
        plt.xlabel('Temps (s)')
        plt.ylabel('Port sélectionné')
        plt.title('Évolution temporelle des choix de port WAN', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3) Graphique comparatif avec poids théoriques
        plt.subplot(2, 3, 3)
        actual_ratios = np.array(counts_clean) / sum(counts_clean) * 100 if sum(counts_clean) > 0 else [0] * len(counts_clean)
        
        # Poids théoriques basés sur la configuration
        theoretical_weights = {1: 3, 2: 2, 3: 1}  # MPLS, Fiber, 4G
        theoretical_ratios = []
        total_weight = sum(theoretical_weights.values())
        
        for port in unique_ports:
            weight = theoretical_weights.get(int(port), 1)
            ratio = weight / total_weight * 100
            theoretical_ratios.append(ratio)
        
        x = np.arange(len(unique_ports))
        width = 0.35
        
        plt.bar(x - width/2, actual_ratios, width, label='Réel', color=colors[:len(unique_ports)], alpha=0.8)
        plt.bar(x + width/2, theoretical_ratios, width, label='Théorique', 
               color='gray', alpha=0.6, edgecolor='black')
        
        plt.xlabel('Chemins WAN')
        plt.ylabel('Pourcentage (%)')
        plt.title('Comparaison Réel vs Théorique', fontsize=14, fontweight='bold')
        plt.xticks(x, labels)
        plt.legend()
        
        # 4) Caractéristiques des liens WAN
        plt.subplot(2, 3, 4)
        link_names = ['MPLS', 'Fiber', '4G']
        bandwidths = [WAN_CONFIGS[name]['bw'] for name in link_names]
        link_colors = ['#2E8B57', '#4169E1', '#FF6347']
        
        plt.bar(range(3), bandwidths, alpha=0.7, color=link_colors)
        plt.ylabel('Bande Passante (Mbps)')
        plt.xlabel('Liens WAN')
        plt.xticks(range(3), link_names)
        plt.title('Capacité des Liens WAN', fontsize=12, fontweight='bold')
        
        # 5) Efficacité de l'équilibrage
        plt.subplot(2, 3, 5)
        if len(actual_ratios) == len(theoretical_ratios):
            efficiency = []
            for actual, theoretical in zip(actual_ratios, theoretical_ratios):
                eff = 100 - abs(actual - theoretical)
                efficiency.append(max(0, eff))
            
            bars = plt.bar(range(len(unique_ports)), efficiency, 
                          color=['green' if e > 80 else 'orange' if e > 60 else 'red' for e in efficiency])
            plt.ylabel('Efficacité (%)')
            plt.xlabel('Chemins WAN')
            plt.title('Efficacité de l\'Équilibrage', fontsize=12, fontweight='bold')
            plt.xticks(range(len(unique_ports)), labels)
            plt.ylim(0, 100)
            
            # Ajout des valeurs sur les barres
            for bar, eff in zip(bars, efficiency):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{eff:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6) Résumé statistique
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        total_flows_count = sum(counts_clean)
        simulation_time = max(data['time']) - min(data['time']) if len(data['time']) > 1 else 0
        avg_flows_per_minute = total_flows_count / (simulation_time / 60) if simulation_time > 0 else 0
        
        stats_text = f"""
STATISTIQUES DE SIMULATION

Durée: {simulation_time:.1f} s
Flux total: {total_flows_count}
Flux/min: {avg_flows_per_minute:.1f}

RÉPARTITION:
"""
        
        # Ajout des stats par port
        for i, port in enumerate(unique_ports):
            port_name = {1: 'MPLS', 2: 'Fiber', 3: '4G'}.get(int(port), f'Port{int(port)}')
            stats_text += f"• {port_name}: {counts_clean[i]} flux ({actual_ratios[i]:.1f}%)\n"
        
        stats_text += f"""
CONFIGURATION WAN:
• MPLS: {WAN_CONFIGS['MPLS']['bw']}Mb/s, {WAN_CONFIGS['MPLS']['delay']}
• Fiber: {WAN_CONFIGS['Fiber']['bw']}Mb/s, {WAN_CONFIGS['Fiber']['delay']}
• 4G: {WAN_CONFIGS['4G']['bw']}Mb/s, {WAN_CONFIGS['4G']['delay']}
        """
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarde
        graph_path = os.path.join(GRAPH_DIR, f'sdwan_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] ✓ Graphiques sauvegardés: {graph_path}")
        
    except Exception as e:
        print(f"[ERROR] Erreur lors de la création des graphiques: {e}")
        import traceback
        traceback.print_exc()
        create_demo_graphs()

def create_demo_graphs():
    """Crée des graphiques de démonstration quand il n'y a pas de données"""
    print("[INFO] Création de graphiques de démonstration...")
    
    plt.figure(figsize=(12, 8))
    
    # Données de démonstration
    demo_ports = [1, 2, 3]
    demo_counts = [30, 20, 10]  # Simulation basée sur les poids 3:2:1
    demo_labels = ['MPLS (Poids: 3)', 'Fiber (Poids: 2)', '4G (Poids: 1)']
    colors = ['#2E8B57', '#4169E1', '#FF6347']
    
    # Graphique 1: Répartition théorique
    plt.subplot(2, 2, 1)
    plt.pie(demo_counts, labels=demo_labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Répartition Théorique du Trafic SD-WAN', fontsize=14, fontweight='bold')
    
    # Graphique 2: Configuration des liens
    plt.subplot(2, 2, 2)
    link_names = ['MPLS', 'Fiber', '4G']
    bandwidths = [20, 100, 10]
    plt.bar(range(3), bandwidths, color=colors, alpha=0.7)
    plt.ylabel('Bande Passante (Mbps)')
    plt.xlabel('Liens WAN')
    plt.xticks(range(3), link_names)
    plt.title('Capacité des Liens WAN', fontsize=12, fontweight='bold')
    
    # Graphique 3: Latence
    plt.subplot(2, 2, 3)
    latencies = [5, 10, 50]
    plt.bar(range(3), latencies, color=colors, alpha=0.7)
    plt.ylabel('Latence (ms)')
    plt.xlabel('Liens WAN')
    plt.xticks(range(3), link_names)
    plt.title('Latence des Liens WAN', fontsize=12, fontweight='bold')
    
    # Graphique 4: Message d'information
    plt.subplot(2, 2, 4)
    plt.axis('off')
    info_text = """
SIMULATION SD-WAN - MODE DÉMONSTRATION

⚠️  Aucune donnée de trafic détectée

CAUSES POSSIBLES:
• Contrôleur Ryu non démarré
• Durée de simulation trop courte
• Problème de connectivité réseau
• Logs non générés

SOLUTIONS:
1. Vérifier que Ryu fonctionne
2. Augmenter SIM_DURATION
3. Vérifier les logs dans /logs/
4. Relancer la simulation

CONFIGURATION ACTUELLE:
• MPLS: 20 Mbps, 5ms (Poids: 3)
• Fiber: 100 Mbps, 10ms (Poids: 2)  
• 4G: 10 Mbps, 50ms (Poids: 1)
    """
    
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Sauvegarde
    graph_path = os.path.join(GRAPH_DIR, f'sdwan_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] ✓ Graphiques de démonstration sauvegardés: {graph_path}")

def main():
    """Fonction principale avec gestion d'erreur améliorée"""
    # Configuration des signaux
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("    SIMULATION SD-WAN CORRIGÉE - ÉQUILIBRAGE DE CHARGE")
    print("=" * 60)
    print()
    print("🔧 VERSION CORRIGÉE avec gestion d'erreurs NaN")
    print()
    print("Configuration:")
    print(f"• {HOSTS_PER_BRANCH} hôtes par succursale")
    print(f"• {len(WAN_CONFIGS)} chemins WAN")
    print(f"• Durée: {SIM_DURATION}s")
    print(f"• Types de trafic: {', '.join(TRAFFIC_TYPES)}")
    print()
    
    setLogLevel('info')
    ensure_dirs()
    
    print("ÉTAPES:")
    print("1. Lancez d'abord le contrôleur Ryu:")
    print("   ryu-manager ryu_sdwan_controller.py")
    print()
    print("2. Ensuite, lancez cette simulation:")
    print("   sudo python3 sdwan_mininet_simulation_fixed.py")
    print()
    print("3. Utilisez Ctrl+C pour arrêter proprement la simulation")
    print()
    
    input("Appuyez sur Entrée pour continuer une fois Ryu démarré...")
    
    try:
        # Lancement de Mininet
        net = launch_mininet()
        print("[INFO] ✓ Topologie créée avec succès")
        
        # 🎨 NOUVEAU: Capture de la topologie
        capture_topology_during_simulation(net)
        
        # Attendre que tous les switches se connectent
        print("[INFO] Connexion des switches au contrôleur...")
        time.sleep(8)
        
        # Démarrage de la simulation de trafic
        traffic_thread = threading.Thread(target=run_multi_traffic_simulation, args=(net,))
        traffic_thread.daemon = True
        traffic_thread.start()
        
        print(f"[INFO] Simulation en cours... (Durée: {SIM_DURATION}s)")
        print("[INFO] Appuyez sur Ctrl+C pour arrêter")
        
        # Attendre la fin de la simulation ou interruption
        try:
            traffic_thread.join(timeout=SIM_DURATION + 20)
        except KeyboardInterrupt:
            pass
        
        print("\n[INFO] Arrêt de la simulation...")
        net.stop()
        
        print("[INFO] Analyse des résultats...")
        time.sleep(3)  # Laisser le temps aux derniers logs
        parse_and_visualize_results()
        
        print("\n" + "=" * 60)
        print("    SIMULATION TERMINÉE")
        print("=" * 60)
        print(f"✓ Logs disponibles dans: {LOG_DIR}/")
        print(f"✓ Statistiques dans: {STATS_DIR}/")
        print(f"✓ Graphiques dans: {GRAPH_DIR}/")
        
    except Exception as e:
        print(f"[ERROR] Erreur durant la simulation: {e}")
        import traceback
        traceback.print_exc()
        
        # Même en cas d'erreur, essayer de générer des graphiques de demo
        try:
            parse_and_visualize_results()
        except:
            create_demo_graphs()
    
    finally:
        # Nettoyage final
        try:
            subprocess.run(['sudo', 'mn', '-c'], capture_output=True)
        except:
            pass

if __name__ == '__main__':
    main()