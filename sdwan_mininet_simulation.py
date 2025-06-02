#!/usr/bin/env python3
###############################################
# File: sdwan_mininet_simulation.py
# Script Mininet pour simuler un SD-WAN avec deux liens WAN, générer du trafic,
# collecter les statistiques et tracer des graphiques à la fin.
###############################################

import os
import time
import logging
from mininet.net import Mininet
from mininet.node import RemoteController, Host
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Répertoire pour stocker les résultats et graphiques
RESULTS_DIR = 'results'
GRAPH_DIR = os.path.join(RESULTS_DIR, 'graphs')
LOG_DIR = 'logs'  # doit correspondre au dossier de l’app Ryu

# Paramètres de la simulation
SIM_DURATION = 20  # durée en secondes pendant laquelle le trafic circule
BW_WAN_1 = 10  # bande passante nominale du lien WAN1 (Mbps)
BW_WAN_2 = 10  # bande passante nominale du lien WAN2 (Mbps)
DELAY_WAN_1 = '10ms'  # délai simulé pour WAN1
DELAY_WAN_2 = '50ms'  # délai simulé pour WAN2
LOSS_WAN_1 = 0  # perte en % pour WAN1
LOSS_WAN_2 = 1  # perte en % pour WAN2 (simule plus de latence/inconfort)

class SDWANTopo(Topo):
    """
    Topologie Mininet :
    - Deux hôtes (h1 et h2) connectés chacun à un switch de succursale (s1, s2).
    - Deux chemins WAN simulés par deux switches intermédiaires (s3, s4) connectés en parallèle.
    - s1 --> s3 (WAN1) --> s2
    - s1 --> s4 (WAN2) --> s2
    - Sur chaque lien on applique des paramètres TCLink (delay, loss, bw).
    """
    def build(self):
        # Ajout des switches
        s1 = self.addSwitch('s1')  # CPE branche A
        s2 = self.addSwitch('s2')  # CPE branche B
        s3 = self.addSwitch('s3')  # Chemin WAN 1
        s4 = self.addSwitch('s4')  # Chemin WAN 2

        # Ajout des hôtes
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')

        # Liens hôtes ↔ CPE
        self.addLink(h1, s1)
        self.addLink(h2, s2)

        # Liens CPE ↔ chemins WAN 1 et 2
        self.addLink(s1, s3,
                     cls=TCLink, bw=BW_WAN_1, delay=DELAY_WAN_1, loss=LOSS_WAN_1)
        self.addLink(s3, s2,
                     cls=TCLink, bw=BW_WAN_1, delay=DELAY_WAN_1, loss=LOSS_WAN_1)

        self.addLink(s1, s4,
                     cls=TCLink, bw=BW_WAN_2, delay=DELAY_WAN_2, loss=LOSS_WAN_2)
        self.addLink(s4, s2,
                     cls=TCLink, bw=BW_WAN_2, delay=DELAY_WAN_2, loss=LOSS_WAN_2)

def ensure_dirs():
    """
    Crée les dossiers de résultats et de graphes si nécessaire.
    """
    for d in [RESULTS_DIR, GRAPH_DIR, LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

def launch_mininet():
    """
    Lance Mininet avec la topologie SD-WAN et le contrôleur Ryu distant.
    """
    topo = SDWANTopo()
    net = Mininet(topo=topo,
                  controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6633),
                  link=TCLink,
                  autoSetMacs=True)
    net.start()
    return net

def run_traffic(net):
    """
    Génère du trafic continu entre h1 et h2 pendant SIM_DURATION secondes.
    Utilise iperf UDP pour créer un trafic mesurable.
    """
    h1 = net.get('h1')
    h2 = net.get('h2')

    # Lance le serveur iperf en UDP sur h2
    h2.cmd('iperf -s -u -i 1 > {}/iperf_server.log &'.format(LOG_DIR))

    # Laisse quelques secondes pour que le serveur démarre
    time.sleep(2)

    # Lance le client iperf en UDP sur h1 pour SIM_DURATION
    # Débit total de 5 Mbps (imposé), on observe comment le controller répartit
    h1.cmd('iperf -c 10.0.0.2 -u -b 5M -t {} -i 1 > {}/iperf_client.log'.format(SIM_DURATION, LOG_DIR))

def parse_logs_and_plot():
    """
    Lit le fichier de log du contrôleur pour déterminer combien de paquets sont passés
    par chaque chemin WAN, puis trace deux graphiques :
      1) Histogramme du nombre de sélections de chaque chemin
      2) Évolution temporelle du choix de chemin (courbe)
    """
    # Lecture du log du contrôleur
    controller_log = os.path.join(LOG_DIR, 'sdwan_controller.log')
    if not os.path.isfile(controller_log):
        print(f\"Fichier de log non trouvé : {controller_log}\")
        return

    ports = []
    timestamps = []
    with open(controller_log, 'r') as f:
        for line in f:
            # On repère les lignes contenant \"Path selected: port=X\"
            if 'Path selected: port=' in line or '--> Path selected: port=' in line:
                try:
                    ts_str = line.split()[1]  # format HH:MM:SS.mmmmmm
                    port = int(line.strip().split('port=')[1])
                    timestamps.append(ts_str)
                    ports.append(port)
                except:
                    continue

    if not ports:
        print(\"Aucune sélection de chemin trouvée dans le log.\")
        return

    # Convertir timestamps en secondes relatives
    # On prend le premier timestamp comme t0
    fmt = '%H:%M:%S.%f'
    import datetime
    t0 = datetime.datetime.strptime(timestamps[0], fmt)
    times_sec = []
    for ts in timestamps:
        t = datetime.datetime.strptime(ts, fmt)
        delta = (t - t0).total_seconds()
        times_sec.append(delta)

    # Assemblage des données
    data = {'port': np.array(ports), 'time': np.array(times_sec)}

    # 1. Histogramme du nombre de sélections par port
    unique_ports = np.unique(data['port'])
    counts = [np.sum(data['port'] == p) for p in unique_ports]

    plt.figure()
    plt.bar([str(p) for p in unique_ports], counts)
    plt.xlabel('Port WAN')
    plt.ylabel('Nombre de sélections')
    plt.title('Répartition du trafic par port WAN')
    histo_path = os.path.join(GRAPH_DIR, 'histogramme_ports.png')
    plt.savefig(histo_path)
    plt.close()
    print(f'Histogramme enregistré : {histo_path}')

    # 2. Courbe évolutionnelle : choix de port au cours du temps
    plt.figure()
    for port in unique_ports:
        mask = (data['port'] == port)
        plt.plot(data['time'][mask], data['port'][mask],
                 'o', label=f'Port {port}', alpha=0.6)
    plt.xlabel('Temps (s)')
    plt.ylabel('Port sélectionné')
    plt.title('Évolution temporelle des choix de port WAN')
    plt.legend()
    curve_path = os.path.join(GRAPH_DIR, 'evolution_ports.png')
    plt.savefig(curve_path)
    plt.close()
    print(f'Courbe d’évolution enregistrée : {curve_path}')

def main():
    setLogLevel('info')
    ensure_dirs()

    print(\"1) Lancez d’abord Ryu :\n   ryu-manager ryu_sdwan_controller.py\n   (attendez qu’il soit prêt)\n\")
    print(\"2) Ensuite, lancez ce script Mininet pour la simulation SD-WAN:\n   sudo python3 sdwan_mininet_simulation.py\n\")

    # On démarre Mininet et on génère le trafic
    net = launch_mininet()
    print(\"Mininet démarré. Attendez quelques instants...\")
    time.sleep(5)  # laisser le temps aux switches de se connecter au controller

    print(f\"Génération de trafic UDP entre h1 et h2 pendant {SIM_DURATION} secondes...\")
    run_traffic(net)

    print(\"Attente de fin de trafic et arrêt de Mininet...\")
    time.sleep(2)
    net.stop()

    print(\"Analyse des logs du contrôleur et génération des graphiques...\")
    parse_logs_and_plot()

if __name__ == '__main__':
    main()
