#!/usr/bin/env python3
###############################################
# File: sdwan_mininet_simulation.py
# Simulation SD-WAN avancée avec multiples hôtes et traffic varié
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

# Configuration
RESULTS_DIR = 'results'
GRAPH_DIR = os.path.join(RESULTS_DIR, 'graphs')
LOG_DIR = 'logs'
STATS_DIR = 'stats'

# Paramètres de simulation
SIM_DURATION = 60  # Durée plus longue pour plus de données
HOSTS_PER_BRANCH = 3  # 3 hôtes par succursale
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
        branch_a_hosts = []
        for i in range(1, HOSTS_PER_BRANCH + 1):
            host = self.addHost(f'h{i}-a', ip=f'10.1.0.{i}/24')
            branch_a_hosts.append(host)
            self.addLink(host, s1)
        
        # Création des hôtes - Branch B  
        branch_b_hosts = []
        for i in range(1, HOSTS_PER_BRANCH + 1):
            host = self.addHost(f'h{i}-b', ip=f'10.2.0.{i}/24')
            branch_b_hosts.append(host)
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
    
    traffic_threads = []
    
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
                traffic_threads.append(thread)
                active_traffics.append((src_name, dst_name))
                
                print(f"[INFO] Nouveau flux démarré à t={current_time:.1f}s")
        
        time.sleep(2)  # Vérification toutes les 2 secondes
    
    print("[INFO] Période de simulation terminée, attente fin des flux...")
    time.sleep(10)  # Laisser les derniers flux se terminer

def parse_and_visualize_results():
    """Analyse des logs et création de graphiques avancés"""
    print("[INFO] Analyse des résultats et génération des graphiques...")
    
    # Lecture des statistiques du contrôleur
    stats_file = os.path.join(STATS_DIR, 'path_statistics.json')
    controller_log = os.path.join(LOG_DIR, 'sdwan_controller.log')
    
    if not os.path.exists(stats_file):
        print(f"[WARNING] Fichier de stats non trouvé: {stats_file}")
        return
    
    # Chargement des données
    with open(stats_file, 'r') as f:
        stats_data = json.load(f)
    
    if not stats_data:
        print("[WARNING] Aucune donnée de statistiques trouvée")
        return
    
    # Extraction des données pour visualisation
    timestamps = []
    path_data = {1: [], 2: [], 3: []}  # MPLS, Fiber, 4G
    
    for entry in stats_data:
        timestamps.append(datetime.fromisoformat(entry['timestamp']))
        for path_id in [1, 2, 3]:
            if str(path_id) in entry['paths']:
                path_data[path_id].append(entry['paths'][str(path_id)]['counter'])
            else:
                path_data[path_id].append(0)
    
    if not timestamps:
        print("[WARNING] Aucune donnée temporelle trouvée")
        return
    
    # Graphique 1: Répartition globale du trafic
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    total_flows = [sum(path_data[p]) for p in [1, 2, 3]]
    labels = ['MPLS\n(Poids: 3)', 'Fiber\n(Poids: 2)', '4G\n(Poids: 1)']
    colors = ['#2E8B57', '#4169E1', '#FF6347']
    
    wedges, texts, autotexts = plt.pie(total_flows, labels=labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    plt.title('Répartition du Trafic par Chemin WAN', fontsize=14, fontweight='bold')
    
    # Graphique 2: Évolution temporelle
    plt.subplot(2, 3, 2)
    time_minutes = [(t - timestamps[0]).total_seconds() / 60 for t in timestamps]
    
    for path_id, label, color in zip([1, 2, 3], ['MPLS', 'Fiber', '4G'], colors):
        plt.plot(time_minutes, path_data[path_id], marker='o', 
                label=label, color=color, linewidth=2, markersize=4)
    
    plt.xlabel('Temps (minutes)')
    plt.ylabel('Nombre de flux')
    plt.title('Évolution du Nombre de Flux par Période', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graphique 3: Histogramme comparatif avec poids théoriques
    plt.subplot(2, 3, 3)
    actual_ratios = np.array(total_flows) / sum(total_flows) * 100 if sum(total_flows) > 0 else [0, 0, 0]
    theoretical_ratios = np.array([3, 2, 1]) / 6 * 100  # Basé sur les poids
    
    x = np.arange(3)
    width = 0.35
    
    plt.bar(x - width/2, actual_ratios, width, label='Réel', color=colors, alpha=0.8)
    plt.bar(x + width/2, theoretical_ratios, width, label='Théorique', 
           color='gray', alpha=0.6, edgecolor='black')
    
    plt.xlabel('Chemins WAN')
    plt.ylabel('Pourcentage (%)')
    plt.title('Comparaison Réel vs Théorique', fontsize=14, fontweight='bold')
    plt.xticks(x, labels)
    plt.legend()
    
    # Graphique 4: Analyse des performances par type de lien
    plt.subplot(2, 3, 4)
    link_names = ['MPLS', 'Fiber', '4G']
    bandwidths = [WAN_CONFIGS[name]['bw'] for name in link_names]
    delays = [int(WAN_CONFIGS[name]['delay'].replace('ms', '')) for name in link_names]
    
    # Graphique à deux axes
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Chemins WAN')
    ax1.set_ylabel('Bande Passante (Mbps)', color=color)
    bars1 = ax1.bar([0, 1, 2], bandwidths, alpha=0.6, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(link_names)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Latence (ms)', color=color)
    line = ax2.plot([0, 1, 2], delays, color=color, marker='o', linewidth=3, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Caractéristiques des Liens WAN')
    
    # Repositionner le subplot
    plt.subplot(2, 3, 4)
    plt.bar(range(3), bandwidths, alpha=0.6, color='tab:blue', label='BP (Mbps)')
    plt.ylabel('Bande Passante (Mbps)')
    plt.xlabel('Chemins WAN')
    plt.xticks(range(3), link_names)
    plt.title('Bande Passante par Lien', fontsize=12, fontweight='bold')
    
    # Graphique 5: Efficacité de l'équilibrage
    plt.subplot(2, 3, 5)
    if sum(total_flows) > 0:
        efficiency = []
        for i, (actual, theoretical) in enumerate(zip(actual_ratios, theoretical_ratios)):
            eff = 100 - abs(actual - theoretical)
            efficiency.append(max(0, eff))
        
        bars = plt.bar(range(3), efficiency, color=['green' if e > 80 else 'orange' if e > 60 else 'red' for e in efficiency])
        plt.ylabel('Efficacité (%)')
        plt.xlabel('Chemins WAN')
        plt.title('Efficacité de l\'Équilibrage', fontsize=12, fontweight='bold')
        plt.xticks(range(3), link_names)
        plt.ylim(0, 100)
        
        # Ajout des valeurs sur les barres
        for bar, eff in zip(bars, efficiency):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{eff:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Graphique 6: Résumé statistique
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Calcul des statistiques
    total_simulation_time = (timestamps[-1] - timestamps[0]).total_seconds() / 60 if len(timestamps) > 1 else 0
    total_flows_count = sum(total_flows)
    avg_flows_per_minute = total_flows_count / total_simulation_time if total_simulation_time > 0 else 0
    
    stats_text = f"""
STATISTIQUES DE SIMULATION

Durée totale: {total_simulation_time:.1f} min
Nombre total de flux: {total_flows_count}
Flux par minute: {avg_flows_per_minute:.1f}

RÉPARTITION:
• MPLS: {total_flows[0]} flux ({actual_ratios[0]:.1f}%)
• Fiber: {total_flows[1]} flux ({actual_ratios[1]:.1f}%)
• 4G: {total_flows[2]} flux ({actual_ratios[2]:.1f}%)

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
    
    # Graphique supplémentaire: Timeline détaillée
    create_detailed_timeline_graph(stats_data)

def create_detailed_timeline_graph(stats_data):
    """Crée un graphique timeline détaillé"""
    plt.figure(figsize=(16, 8))
    
    timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in stats_data]
    time_minutes = [(t - timestamps[0]).total_seconds() / 60 for t in timestamps]
    
    # Données cumulatives
    cumulative_data = {1: [], 2: [], 3: []}
    running_totals = {1: 0, 2: 0, 3: 0}
    
    for entry in stats_data:
        for path_id in [1, 2, 3]:
            if str(path_id) in entry['paths']:
                running_totals[path_id] += entry['paths'][str(path_id)]['counter']
            cumulative_data[path_id].append(running_totals[path_id])
    
    # Graphique en aires empilées
    colors = ['#2E8B57', '#4169E1', '#FF6347']
    labels = ['MPLS (Poids: 3)', 'Fiber (Poids: 2)', '4G (Poids: 1)']
    
    plt.stackplot(time_minutes, 
                 cumulative_data[1], cumulative_data[2], cumulative_data[3],
                 labels=labels, colors=colors, alpha=0.7)
    
    plt.xlabel('Temps (minutes)', fontsize=12)
    plt.ylabel('Flux Cumulés', fontsize=12)
    plt.title('Évolution Cumulative du Trafic par Chemin WAN', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Ajout d'annotations pour les périodes importantes
    if len(time_minutes) > 10:
        mid_point = len(time_minutes) // 2
        plt.annotate('Pic d\'activité', 
                    xy=(time_minutes[mid_point], sum(cumulative_data[p][mid_point] for p in [1,2,3])),
                    xytext=(time_minutes[mid_point] + 2, sum(cumulative_data[p][mid_point] for p in [1,2,3]) + 10),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    plt.tight_layout()
    timeline_path = os.path.join(GRAPH_DIR, f'timeline_cumulative_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] ✓ Timeline détaillée sauvegardée: {timeline_path}")

def main():
    """Fonction principale"""
    # Configuration des signaux
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("    SIMULATION SD-WAN AVANCÉE - ÉQUILIBRAGE DE CHARGE")
    print("=" * 60)
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
    print("   sudo python3 sdwan_mininet_simulation.py")
    print()
    print("3. Utilisez Ctrl+C pour arrêter proprement la simulation")
    print()
    
    input("Appuyez sur Entrée pour continuer une fois Ryu démarré...")
    
    try:
        # Lancement de Mininet
        net = launch_mininet()
        print("[INFO] ✓ Topologie créée avec succès")
        
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
    
    finally:
        # Nettoyage final
        try:
            subprocess.run(['sudo', 'mn', '-c'], capture_output=True)
        except:
            pass

if __name__ == '__main__':
    main()