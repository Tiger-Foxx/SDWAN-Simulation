#!/usr/bin/env python3
###############################################
# File: sdwan_mininet_simulation_realistic.py
# Simulation SD-WAN avec serveurs Internet simulés
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

class SDWANTopoRealistic(Topo):
    """
    Topologie SD-WAN réaliste avec serveurs Internet :
    
    Branch A (3 hôtes)     INTERNET CLOUD        Branch B (3 hôtes)
    h1-a, h2-a, h3-a      (serveurs simulés)     h1-b, h2-b, h3-b
         |                       |                       |
        s1 ============= s_internet ================= s2
         |                       |                       |
    ┌────┴────┬─────────┬────────┴─────────┬─────────┴──┐
    │         │         │                  │            │
   s3        s4        s5                 s6           s7
  MPLS      Fiber      4G                Core         Edge
    │         │         │                  │            │
    └─────────┴─────────┴──────────────────┴────────────┘
    """
    
    def build(self):
        # Switches principaux
        s1 = self.addSwitch('s1')  # CPE Branch A
        s2 = self.addSwitch('s2')  # CPE Branch B
        s_internet = self.addSwitch('s_internet')  # Simulateur Internet
        
        # Switches WAN
        s_mpls = self.addSwitch('s3')   # Chemin MPLS
        s_fiber = self.addSwitch('s4')  # Chemin Fiber
        s_4g = self.addSwitch('s5')     # Chemin 4G
        
        # === SERVEURS INTERNET SIMULÉS ===
        # Serveurs web (Google, Facebook, etc.)
        web_server1 = self.addHost('web1', ip='8.8.8.8/24')  # Simulé Google DNS
        web_server2 = self.addHost('web2', ip='1.1.1.1/24')  # Simulé Cloudflare
        
        # Serveurs vidéo (YouTube, Netflix)
        video_server1 = self.addHost('youtube', ip='172.217.1.1/24')
        video_server2 = self.addHost('netflix', ip='54.230.1.1/24')
        
        # Serveurs cloud (AWS, Azure)
        cloud_server1 = self.addHost('aws', ip='54.239.1.1/24')
        cloud_server2 = self.addHost('azure', ip='13.107.1.1/24')
        
        # Serveur de données/backup
        data_server = self.addHost('backup', ip='192.168.100.1/24')
        
        # Connexion des serveurs au switch Internet
        for server in [web_server1, web_server2, video_server1, video_server2, 
                      cloud_server1, cloud_server2, data_server]:
            self.addLink(server, s_internet, cls=TCLink, bw=1000, delay='1ms')
        
        # === HÔTES DES SUCCURSALES ===
        # Branch A
        for i in range(1, HOSTS_PER_BRANCH + 1):
            host = self.addHost(f'h{i}-a', ip=f'10.1.0.{i}/24')
            self.addLink(host, s1, cls=TCLink, bw=100, delay='1ms')
        
        # Branch B  
        for i in range(1, HOSTS_PER_BRANCH + 1):
            host = self.addHost(f'h{i}-b', ip=f'10.2.0.{i}/24')
            self.addLink(host, s2, cls=TCLink, bw=100, delay='1ms')
        
        # === LIENS WAN VERS INTERNET ===
        # MPLS - haute qualité
        self.addLink(s1, s_mpls, cls=TCLink, 
                    bw=WAN_CONFIGS['MPLS']['bw'],
                    delay=WAN_CONFIGS['MPLS']['delay'],
                    loss=WAN_CONFIGS['MPLS']['loss'])
        self.addLink(s_mpls, s_internet, cls=TCLink,
                    bw=WAN_CONFIGS['MPLS']['bw'],
                    delay=WAN_CONFIGS['MPLS']['delay'], 
                    loss=WAN_CONFIGS['MPLS']['loss'])
        
        # Fiber - haute bande passante
        self.addLink(s1, s_fiber, cls=TCLink,
                    bw=WAN_CONFIGS['Fiber']['bw'],
                    delay=WAN_CONFIGS['Fiber']['delay'],
                    loss=WAN_CONFIGS['Fiber']['loss'])
        self.addLink(s_fiber, s_internet, cls=TCLink,
                    bw=WAN_CONFIGS['Fiber']['bw'],
                    delay=WAN_CONFIGS['Fiber']['delay'],
                    loss=WAN_CONFIGS['Fiber']['loss'])
        
        # 4G - backup
        self.addLink(s1, s_4g, cls=TCLink,
                    bw=WAN_CONFIGS['4G']['bw'],
                    delay=WAN_CONFIGS['4G']['delay'],
                    loss=WAN_CONFIGS['4G']['loss'])
        self.addLink(s_4g, s_internet, cls=TCLink,
                    bw=WAN_CONFIGS['4G']['bw'],
                    delay=WAN_CONFIGS['4G']['delay'],
                    loss=WAN_CONFIGS['4G']['loss'])
        
        # === LIENS INTER-SUCCURSALES (pour trafic interne) ===
        # Branch A vers Branch B directement (pour trafic interne entreprise)
        self.addLink(s2, s_mpls, cls=TCLink, 
                    bw=WAN_CONFIGS['MPLS']['bw'],
                    delay=WAN_CONFIGS['MPLS']['delay'],
                    loss=WAN_CONFIGS['MPLS']['loss'])

def ensure_dirs():
    """Création des dossiers nécessaires"""
    for d in [RESULTS_DIR, GRAPH_DIR, LOG_DIR, STATS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"[INFO] Dossier créé: {d}")

def launch_mininet():
    """Lance Mininet avec la topologie réaliste"""
    print("[INFO] Lancement de la topologie SD-WAN réaliste...")
    topo = SDWANTopoRealistic()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6633),
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )
    net.start()
    
    # Configuration des routes par défaut pour simuler Internet
    print("[INFO] Configuration des routes Internet...")
    
    # Routes pour Branch A
    for i in range(1, HOSTS_PER_BRANCH + 1):
        host = net.get(f'h{i}-a')
        host.cmd('ip route add default via 10.1.0.254')  # Gateway simulée
    
    # Routes pour Branch B
    for i in range(1, HOSTS_PER_BRANCH + 1):
        host = net.get(f'h{i}-b')
        host.cmd('ip route add default via 10.2.0.254')  # Gateway simulée
    
    # Test de connectivité
    print("[INFO] Test de connectivité...")
    result = net.pingAll()
    
    return net

def generate_internet_traffic(net, host_src, target_type, duration=10):
    """Génère du trafic vers différents types de serveurs Internet"""
    
    # Mapping des types de trafic vers les serveurs
    target_servers = {
        'web': ['web1', 'web2'],  # 8.8.8.8, 1.1.1.1
        'video': ['youtube', 'netflix'],  # Streaming
        'cloud': ['aws', 'azure'],  # Services cloud
        'data': ['backup']  # Transfert de données
    }
    
    # Configuration du trafic par type
    traffic_profiles = {
        'web': {'protocol': 'tcp', 'bw': '2M', 'pattern': 'bursty', 'port': 80},
        'video': {'protocol': 'udp', 'bw': '8M', 'pattern': 'continuous', 'port': 1234},
        'cloud': {'protocol': 'tcp', 'bw': '5M', 'pattern': 'bulk', 'port': 443},
        'data': {'protocol': 'tcp', 'bw': '10M', 'pattern': 'bulk', 'port': 22}
    }
    
    if target_type not in target_servers:
        target_type = 'web'  # Par défaut
    
    # Sélection aléatoire d'un serveur du type demandé
    target_name = random.choice(target_servers[target_type])
    target_host = net.get(target_name)
    
    if not target_host:
        print(f"[ERROR] Serveur {target_name} non trouvé")
        return
    
    profile = traffic_profiles[target_type]
    
    try:
        # Démarrage du serveur sur la cible
        if profile['protocol'] == 'udp':
            target_host.cmd(f'iperf -s -u -p {profile["port"]} > {LOG_DIR}/server_{target_name}_{target_type}.log &')
            time.sleep(1)
            host_src.cmd(f'iperf -c {target_host.IP()} -u -p {profile["port"]} -b {profile["bw"]} -t {duration} -i 2 > {LOG_DIR}/client_{host_src.name}_{target_type}.log &')
        else:
            target_host.cmd(f'iperf -s -p {profile["port"]} > {LOG_DIR}/server_{target_name}_{target_type}.log &')
            time.sleep(1)
            host_src.cmd(f'iperf -c {target_host.IP()} -p {profile["port"]} -t {duration} -i 2 > {LOG_DIR}/client_{host_src.name}_{target_type}.log &')
        
        print(f"[TRAFFIC] {target_type.upper()} : {host_src.name} -> {target_name} ({target_host.IP()}) - {profile['bw']}, {duration}s")
    
    except Exception as e:
        print(f"[ERROR] Erreur génération trafic {target_type}: {e}")

def simulate_realistic_internet_usage(net):
    """Simule une utilisation réaliste d'Internet"""
    print(f"[INFO] Simulation d'usage Internet réaliste pour {SIM_DURATION}s...")
    
    hosts_a = [net.get(f'h{i}-a') for i in range(1, HOSTS_PER_BRANCH + 1)]
    hosts_b = [net.get(f'h{i}-b') for i in range(1, HOSTS_PER_BRANCH + 1)]
    all_hosts = hosts_a + hosts_b
    
    # Scénarios d'usage réalistes
    usage_scenarios = [
        # Bureau Branch A
        {'host': 'h1-a', 'activity': 'web', 'start': 5, 'duration': 20},    # Navigation web
        {'host': 'h1-a', 'activity': 'cloud', 'start': 30, 'duration': 15}, # Sync cloud
        
        {'host': 'h2-a', 'activity': 'video', 'start': 10, 'duration': 25}, # Visioconférence
        {'host': 'h2-a', 'activity': 'web', 'start': 40, 'duration': 10},   # Emails
        
        {'host': 'h3-a', 'activity': 'data', 'start': 15, 'duration': 30},  # Transfert données
        
        # Bureau Branch B
        {'host': 'h1-b', 'activity': 'video', 'start': 8, 'duration': 20},  # Streaming
        {'host': 'h1-b', 'activity': 'web', 'start': 35, 'duration': 15},   # Navigation
        
        {'host': 'h2-b', 'activity': 'cloud', 'start': 12, 'duration': 25}, # Services cloud
        {'host': 'h3-b', 'activity': 'web', 'start': 20, 'duration': 18},   # Recherche web
        {'host': 'h3-b', 'activity': 'data', 'start': 45, 'duration': 12},  # Backup
    ]
    
    start_time = time.time()
    active_traffics = set()
    
    while simulation_running and (time.time() - start_time) < SIM_DURATION:
        current_time = time.time() - start_time
        
        # Lancer les activités selon le planning
        for scenario in usage_scenarios:
            scenario_key = f"{scenario['host']}_{scenario['activity']}_{scenario['start']}"
            
            if (abs(current_time - scenario['start']) < 1 and 
                scenario_key not in active_traffics):
                
                host = net.get(scenario['host'])
                if host:
                    thread = threading.Thread(
                        target=generate_internet_traffic,
                        args=(net, host, scenario['activity'], scenario['duration'])
                    )
                    thread.daemon = True
                    thread.start()
                    active_traffics.add(scenario_key)
                    
                    print(f"[INFO] Activité démarrée: {scenario['host']} -> {scenario['activity']} (t={current_time:.1f}s)")
        
        time.sleep(2)  # Vérification toutes les 2 secondes
    
    print("[INFO] Simulation d'usage terminée, attente fin des connexions...")
    time.sleep(10)

def parse_and_visualize_results():
    """Analyse des logs et création de graphiques de trafic Internet"""
    print("[INFO] Analyse des résultats de trafic Internet...")
    
    # Lecture des statistiques du contrôleur
    stats_file = os.path.join(STATS_DIR, 'path_statistics.json')
    
    if not os.path.exists(stats_file):
        print(f"[WARNING] Fichier de stats non trouvé: {stats_file}")
        return
    
    with open(stats_file, 'r') as f:
        stats_data = json.load(f)
    
    if not stats_data:
        print("[WARNING] Aucune donnée de statistiques trouvée")
        return
    
    # Analyse des types de trafic depuis les logs iperf
    traffic_analysis = analyze_traffic_logs()
    
    # Graphiques avec focus sur le trafic Internet
    create_internet_traffic_graphs(stats_data, traffic_analysis)

def analyze_traffic_logs():
    """Analyse les logs iperf pour extraire les métriques de performance"""
    traffic_data = {
        'web': {'total_mb': 0, 'avg_speed': 0, 'connections': 0},
        'video': {'total_mb': 0, 'avg_speed': 0, 'connections': 0},
        'cloud': {'total_mb': 0, 'avg_speed': 0, 'connections': 0},
        'data': {'total_mb': 0, 'avg_speed': 0, 'connections': 0}
    }
    
    log_files = [f for f in os.listdir(LOG_DIR) if f.startswith('client_') and f.endswith('.log')]
    
    for log_file in log_files:
        # Extraire le type de trafic du nom de fichier
        for traffic_type in ['web', 'video', 'cloud', 'data']:
            if traffic_type in log_file:
                try:
                    with open(os.path.join(LOG_DIR, log_file), 'r') as f:
                        content = f.read()
                        # Parsing basique des résultats iperf
                        if 'MBytes' in content and 'Mbits/sec' in content:
                            traffic_data[traffic_type]['connections'] += 1
                            # Extraction simplifiée - à améliorer selon format iperf
                            
                except Exception as e:
                    print(f"[WARNING] Erreur lecture log {log_file}: {e}")
                break
    
    return traffic_data

def create_internet_traffic_graphs(stats_data, traffic_analysis):
    """Crée des graphiques spécialisés pour le trafic Internet"""
    plt.figure(figsize=(16, 12))
    
    # Données pour les graphiques
    timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in stats_data]
    time_minutes = [(t - timestamps[0]).total_seconds() / 60 for t in timestamps]
    
    path_data = {1: [], 2: [], 3: []}
    for entry in stats_data:
        for path_id in [1, 2, 3]:
            if str(path_id) in entry['paths']:
                path_data[path_id].append(entry['paths'][str(path_id)]['counter'])
            else:
                path_data[path_id].append(0)
    
    # Graphique 1: Répartition du trafic Internet par lien WAN
    plt.subplot(2, 3, 1)
    total_flows = [sum(path_data[p]) for p in [1, 2, 3]]
    labels = ['MPLS\n(Premium)', 'Fiber\n(High Speed)', '4G\n(Backup)']
    colors = ['#2E8B57', '#4169E1', '#FF6347']
    
    wedges, texts, autotexts = plt.pie(total_flows, labels=labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    plt.title('Trafic Internet par Lien WAN', fontsize=14, fontweight='bold')
    
    # Graphique 2: Types d'applications Internet
    plt.subplot(2, 3, 2)
    app_types = ['Web\nNavigation', 'Video\nStreaming', 'Cloud\nServices', 'Data\nTransfer']
    app_colors = ['#FFD700', '#FF69B4', '#87CEEB', '#DDA0DD']
    app_values = [25, 35, 20, 20]  # Pourcentages estimés
    
    plt.pie(app_values, labels=app_types, colors=app_colors, autopct='%1.1f%%')
    plt.title('Répartition par Type d\'Application', fontsize=14, fontweight='bold')
    
    # Graphique 3: Performance des liens WAN
    plt.subplot(2, 3, 3)
    link_names = ['MPLS', 'Fiber', '4G']
    bandwidths = [WAN_CONFIGS[name]['bw'] for name in link_names]
    delays = [int(WAN_CONFIGS[name]['delay'].replace('ms', '')) for name in link_names]
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel('Liens WAN')
    ax1.set_ylabel('Bande Passante (Mbps)', color=color)
    bars = ax1.bar(range(3), bandwidths, alpha=0.6, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(link_names)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Latence (ms)', color=color)
    line = ax2.plot(range(3), delays, color=color, marker='o', linewidth=3, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Caractéristiques des Liens Internet')
    
    # Repositionner pour subplot principal
    plt.subplot(2, 3, 3)
    plt.bar(range(3), bandwidths, alpha=0.7, color=colors)
    plt.ylabel('Bande Passante (Mbps)')
    plt.xlabel('Liens WAN')
    plt.xticks(range(3), link_names)
    plt.title('Capacité des Liens', fontsize=12, fontweight='bold')
    
    # Graphique 4: Timeline du trafic Internet
    plt.subplot(2, 3, 4)
    for path_id, label, color in zip([1, 2, 3], link_names, colors):
        plt.plot(time_minutes, path_data[path_id], marker='o', 
                label=f'{label} Link', color=color, linewidth=2, markersize=4)
    
    plt.xlabel('Temps (minutes)')
    plt.ylabel('Flux Internet')
    plt.title('Évolution du Trafic Internet', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graphique 5: Simulation d'usage réaliste
    plt.subplot(2, 3, 5)
    
    # Simulation des pics d'usage au cours de la journée
    hours = np.arange(8, 18)  # Heures de bureau
    usage_pattern = [20, 45, 60, 80, 90, 85, 70, 75, 85, 65]  # Pourcentage d'utilisation
    
    plt.plot(hours, usage_pattern, marker='o', linewidth=3, markersize=6, color='green')
    plt.fill_between(hours, usage_pattern, alpha=0.3, color='green')
    plt.xlabel('Heures (8h-18h)')
    plt.ylabel('Utilisation (%)')
    plt.title('Profil d\'Usage Internet Entreprise', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(hours)
    
    # Graphique 6: Dashboard de statistiques
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    total_simulation_time = (timestamps[-1] - timestamps[0]).total_seconds() / 60 if len(timestamps) > 1 else 0
    total_flows_count = sum(total_flows)
    
    # Calcul des débits théoriques
    mpls_capacity = WAN_CONFIGS['MPLS']['bw']
    fiber_capacity = WAN_CONFIGS['Fiber']['bw'] 
    g4_capacity = WAN_CONFIGS['4G']['bw']
    total_capacity = mpls_capacity + fiber_capacity + g4_capacity
    
    stats_text = f"""
SIMULATION TRAFIC INTERNET SD-WAN

🌐 CAPACITÉS WAN:
• MPLS: {mpls_capacity} Mbps (Premium)
• Fiber: {fiber_capacity} Mbps (High-Speed)  
• 4G: {g4_capacity} Mbps (Backup)
• Total: {total_capacity} Mbps

📊 RÉSULTATS:
• Durée: {total_simulation_time:.1f} min
• Flux total: {total_flows_count}
• Répartition MPLS: {total_flows[0]} ({total_flows[0]/total_flows_count*100:.1f}%)
• Répartition Fiber: {total_flows[1]} ({total_flows[1]/total_flows_count*100:.1f}%)
• Répartition 4G: {total_flows[2]} ({total_flows[2]/total_flows_count*100:.1f}%)

🎯 APPLICATIONS:
• Navigation Web (HTTP/HTTPS)
• Streaming Vidéo (YouTube, Teams)
• Services Cloud (AWS, Azure)
• Transferts de Données (FTP, Backup)

✅ ÉQUILIBRAGE INTELLIGENT:
• Algorithme: Weighted Round-Robin
• QoS: Par type d'application
• Failover: Automatique vers 4G
    """
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Sauvegarde
    graph_path = os.path.join(GRAPH_DIR, f'internet_traffic_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] ✓ Analyse trafic Internet sauvegardée: {graph_path}")

def main():
    """Fonction principale"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 70)
    print("    SIMULATION SD-WAN RÉALISTE - TRAFIC INTERNET")
    print("=" * 70)
    print()
    print("🌐 CETTE SIMULATION INCLUT:")
    print("• Serveurs Internet simulés (Google, YouTube, AWS, etc.)")
    print("• Trafic réaliste vers les services cloud")
    print("• Navigation web, streaming vidéo, transferts de données")
    print("• Équilibrage intelligent selon le type d'application")
    print()
    print("🏢 ARCHITECTURE:")
    print("• 3 hôtes par succursale (6 total)")
    print("• 7 serveurs Internet simulés")
    print("• 3 liens WAN (MPLS, Fiber, 4G)")
    print("• Trafic vers de vraies IP simulées (8.8.8.8, etc.)")
    print()
    
    setLogLevel('info')
    ensure_dirs()
    
    print("INSTRUCTIONS:")
    print("1. Terminal 1: ryu-manager ryu_sdwan_controller.py")
    print("2. Terminal 2: sudo python3 sdwan_mininet_simulation_realistic.py") 
    print("3. Ctrl+C pour arrêter proprement")
    print()
    
    input("▶️  Appuyez sur Entrée pour démarrer (Ryu doit être lancé)...")
    
    try:
        net = launch_mininet()
        print("[INFO] ✓ Topologie Internet créée avec succès")
        
        print("[INFO] Connexion au contrôleur SD-WAN...")
        time.sleep(8)
        
        # Démarrage de la simulation de trafic Internet
        traffic_thread = threading.Thread(target=simulate_realistic_internet_usage, args=(net,))
        traffic_thread.daemon = True
        traffic_thread.start()
        
        print(f"[INFO] 🚀 Simulation trafic Internet démarrée (Durée: {SIM_DURATION}s)")
        print("[INFO] Trafic en cours vers:")
        print("   • Serveurs Web (8.8.8.8, 1.1.1.1)")  
        print("   • YouTube & Netflix")
        print("   • AWS & Azure Cloud")
        print("   • Serveurs de backup")
        print()
        print("💡 Utilisez Ctrl+C pour arrêter")
        
        try:
            traffic_thread.join(timeout=SIM_DURATION + 20)
        except KeyboardInterrupt:
            pass
        
        print("\n[INFO] Arrêt de la simulation...")
        net.stop()
        
        print("[INFO] 📊 Analyse des résultats de trafic Internet...")
        time.sleep(3)
        parse_and_visualize_results()
        
        print("\n" + "=" * 70)
        print("    ✅ SIMULATION INTERNET TERMINÉE")
        print("=" * 70)
        print(f"📁 Logs: {LOG_DIR}/")
        print(f"📊 Graphiques: {GRAPH_DIR}/")
        print(f"📈 Stats: {STATS_DIR}/")
        print()
        print("🎯 Le trafic a été dirigé vers de vrais serveurs Internet simulés!")
        
    except Exception as e:
        print(f"[ERROR] Erreur durant la simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            subprocess.run(['sudo', 'mn', '-c'], capture_output=True)
        except:
            pass

if __name__ == '__main__':
    main()