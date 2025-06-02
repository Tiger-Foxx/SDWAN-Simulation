# Simulation SD-WAN avec Mininet et Ryu

## Contexte et justification du projet

Dans le cadre d’un travail académique, le thème porte sur **l’équilibrage de charge dans un réseau SD-WAN**. Plusieurs points de contexte sont essentiels pour comprendre la finalité et la méthodologie employée :

1. **SDN vs SD-WAN**

   * **SDN (Software-Defined Networking)** désigne une architecture réseau où le plan de contrôle est découplé du plan de données. Le contrôleur SDN (par exemple Ryu) gère dynamiquement les flux sur des liens WAN (MPLS, VPN, 4G, fibre optique, etc.), mais son usage n’est pas spécifiquement orienté SD-WAN.
   * **SD-WAN (Software-Defined WAN)** se concentre sur l’optimisation, l’équilibrage de charge et la gestion de qualité de service (QoS) entre plusieurs liens WAN hétérogènes (MPLS, Internet public, 4G/5G). L’objectif principal est de garantir qu’une entreprise utilise ses différents liens WAN de manière optimale selon des critères comme la bande passante disponible, la latence, la fiabilité, ou la priorité de certains services.

2. **Problématique ciblée**

   * En production, une entreprise possède souvent plusieurs connexions WAN. Un SD-WAN doit pouvoir distribuer le trafic au travers de ces liaisons selon des règles préétablies (par exemple, priorité aux applications critiques sur un lien réputé plus fiable, routage automatique en cas de défaillance, ou simple partages pondérés de la bande passante).
   * L’équilibrage de charge dans un SD-WAN peut se fonder sur plusieurs critères : poids (pour donner plus de trafic à un lien), latence simulée, perte de paquets, ou encore classification du trafic (vidéo, VoIP, web, etc.).

3. **Objectif de la simulation**

   * Développer une maquette fonctionnelle de SD-WAN à l’aide de Mininet (simulateur de réseau) et de Ryu (contrôleur SDN OpenFlow v1.3).
   * Implémenter un algorithme d’équilibrage de charge *weighted round-robin* (round-robin pondéré) qui décide dynamiquement, à chaque détection de nouveau flux, quel lien WAN exploiter.
   * Simuler la latence et la bande passante différentes de chaque lien WAN via `TCLink` dans Mininet (paramètres `bw`, `delay`, `loss`).
   * Générer du trafic (avec `iperf` en UDP) entre deux sites distants et collecter les logs d’élection de chemin pour produire des graphiques finaux présentant la répartition du trafic et l’évolution temporelle de l’équilibrage.

4. **Pourquoi cette approche ?**

   * Mininet permet de construire une topologie à moindres frais (tous les éléments tournent en local).
   * Ryu est un contrôleur python OpenFlow 1.3 relativement simple à programmer, ce qui facilite l’implémentation d’un algorithme d’équilibrage sur des liens WAN simulés.
   * L’usage de création automatique de graphes en Python (avec matplotlib) permet de produire des résultats visuels exploitables pour valider que le poids des chemins WAN a été respecté.

---

## Contenu du répertoire et structure

```
projet_sdwan/            # Dossier racine du projet
├── ryu_sdwan_controller.py   # Contrôleur Ryu implémentant le weighted RR
├── sdwan_mininet_simulation.py  # Script Mininet de simulation + génération de trafic + graphes
├── logs/                    # Dossier où Ryu et iperf stockent leurs logs
│   ├── sdwan_controller.log  # Logs des sélections de chemin WAN par Ryu
│   ├── iperf_server.log      # Logs du serveur iperf (h2)
│   └── iperf_client.log      # Logs du client iperf (h1)
├── results/                 # Résultats finaux (logs parsés, CSV, etc.)
│   └── graphs/              # Graphiques générés
│       ├── histogramme_ports.png
│       └── evolution_ports.png
└── README.md                # Ce document de description du projet
```

* **`ryu_sdwan_controller.py`** : contiendra la logique OpenFlow pour

  * Apprendre les adresses MAC (learning)
  * Détecter les nouveaux flux (PacketIn)
  * Exécuter un algorithme pondéré de round-robin pour déterminer quel port (lien WAN) utiliser
  * Journaliser chaque sélection de chemin dans `logs/sdwan_controller.log`

* **`sdwan_mininet_simulation.py`** :

  * Construire une topologie Mininet à deux liens WAN
  * Lancer le trafic UDP avec `iperf` (5 Mbps) entre deux hôtes h1 et h2
  * Arrêter Mininet après `SIM_DURATION` secondes
  * Parcourir les logs Ryu pour extraire les moments où un chemin a été sélectionné
  * Générer deux graphiques (histogramme et évolution temporelle)

* **`logs/`** : Regroupe tous les fichiers de logs générés durant la simulation.

  * `sdwan_controller.log` contient des lignes du type `Path selected: port=X` avec horodatage.
  * `iperf_server.log` et `iperf_client.log` enregistrent la performance du trafic UDP.

* **`results/graphs/`** :

  * `histogramme_ports.png` : Nombre total de sélections par port WAN.
  * `evolution_ports.png` : Représentation temporelle des sélections de port (chaque point = une décision).

---

## Prérequis et installation

1. **Système d’exploitation Linux** (Ubuntu 18.04+ recommandé, ou distribution similaire).
2. **Mininet** (version \~2.3)

   * Tutoriel détaillé pour installer Mininet : [https://github.com/mininet/mininet/wiki/Quick-Start](https://github.com/mininet/mininet/wiki/Quick-Start)
   * Exemple d’installation rapide sur Ubuntu :

     ```bash
     sudo apt-get update
     sudo apt-get install -y build-essential git python3-pip
     git clone https://github.com/mininet/mininet.git
     mininet/util/install.sh -a   # installe Mininet, Open vSwitch, etc.
     ```
3. **Ryu Controller** (OpenFlow v1.3)

   * Installer via pip :

     ```bash
     sudo pip3 install ryu
     ```
   * Vérifier la version installée :

     ```bash
     ryu-manager --version
     ```
4. **Python 3.6+**

   * Bibliothèques nécessaires :

     ```bash
     sudo apt-get install -y python3-numpy python3-matplotlib
     ```
5. **Autorisation root**

   * Mininet doit être lancé en `sudo` pour créer les interfaces et configurer TCLink.

---

## Explication détaillée du code

### 1. Contrôleur SD-WAN : `ryu_sdwan_controller.py`

```python
###############################################
# Fichier : ryu_sdwan_controller.py
# Contrôleur Ryu pour équilibrage de charge SD-WAN pondéré
###############################################

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
from ryu.lib import hub
import logging
import time
import os

# 1. Configuration des chemins WAN (port -> poids)
PATH_WEIGHTS = {1: 1, 2: 2}

# 2. Répertoire et fichier de log
LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'sdwan_controller.log')

class SDWANController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SDWANController, self).__init__(*args, **kwargs)

        # Création du dossier de logs si nécessaire
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        logging.basicConfig(
            filename=LOG_FILE,
            format='%(asctime)s %(message)s',
            level=logging.INFO
        )

        # Table MAC → port
        self.mac_to_port = {}

        # Initialisation des compteurs pour chaque port WAN
        self.path_counters = {}
        for port, weight in PATH_WEIGHTS.items():
            self.path_counters[port] = {'weight': weight, 'counter': 0}

        # Thread vert pour réinitialiser périodiquement les compteurs
        self.reset_interval = 10  # en secondes
        self.monitor_thread = hub.spawn(self._reset_counters)

    def _reset_counters(self):
        """
        Réinitialise tous les compteurs à 0 tous les reset_interval
        """
        while True:
            hub.sleep(self.reset_interval)
            for port in self.path_counters:
                self.path_counters[port]['counter'] = 0
            logging.info("‼ Counters reset pour tous les chemins WAN")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        Lorsqu’un switch se connecte au contrôleur, on installe la table-miss
        pour envoyer au contrôleur tout paquet qui n’a pas de flow.
        """
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Match qui capture tout (table-miss)
        match = parser.OFPMatch()
        # Action : envoyée au contrôleur
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        # Priorité 0 pour la table-miss
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """
        Ajoute une entrée de flux dans le switch.
        - datapath : instance du switch
        - priority : priorité du flow
        - match : critère de correspondance
        - actions : liste d’actions à exécuter (ici, output vers port)
        - buffer_id : si le switch a mis le paquet en buffer, on l’utilise pour fast-path
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    def select_path(self):
        """
        Algorithme de sélection pondérée (weighted_round_robin) :
        - Pour chaque port, on calcule ratio = counter / weight.
        - On choisit le port avec le plus petit ratio.
        - On incrémente ensuite son compteur.
        """
        best_port = None
        best_ratio = None
        for port, data in self.path_counters.items():
            weight = data['weight']
            counter = data['counter']
            ratio = counter / float(weight)
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_port = port
        self.path_counters[best_port]['counter'] += 1
        return best_port

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
        Gère l’événement PacketIn :
        - Apprentissage MAC → port
        - Si destiné localement, forward normalement
        - Sinon, appel à select_path() pour choisir un lien WAN
        - Installation d’un flow pour éviter des futurs PacketIn similaires
        - Log de la sélection de port
        """
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        # Apprentissage de la source
        self.mac_to_port[dpid][src] = in_port

        # Si on connaît déjà le port de destination en interne → on forward
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            # Sinon, on choisit un chemin WAN via weighted RR
            out_port = self.select_path()
            logging.info(f"--> Path selected: port={out_port}")

        actions = [parser.OFPActionOutput(out_port)]

        # Installation du flow pour ce couple (src, dst) afin d’éviter futurs PacketIn
        if msg.buffer_id != ofproto.OFP_NO_BUFFER:
            match = parser.OFPMatch(eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions, msg.buffer_id)
            return

        match = parser.OFPMatch(eth_dst=dst, eth_src=src)
        self.add_flow(datapath, 1, match, actions)
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=msg.buffer_id,
                                  in_port=in_port,
                                  actions=actions,
                                  data=data)
        datapath.send_msg(out)
```

#### Détails et points clés

* **`PATH_WEIGHTS`** : dictionnaire qui associe chaque port de sortie (1 → WAN1, 2 → WAN2) à un poids. Si on veut dupliquer un lien ou en ajouter d’autres, il suffit d’ajouter des entrées : `{1:1, 2:2, 3:3, ...}`.
* **Apprentissage MAC (learning switch)** :

  * `self.mac_to_port[dpid][src] = in_port` associe l’adresse MAC source au port d’entrée.
  * Si la destination est connue (à l’intérieur du département local), on forward directement sans passer par SD-WAN (on pourrait adapter pour traiter différemment le trafic local).
* **Sélection de chemin : `select_path()`** :

  1. Calcule, pour chaque port, le ratio `counter / weight`.
  2. Le port avec le plus petit ratio est choisi.
  3. On incrémente son compteur.
  4. À chaque nouvel arrivage de flux inconnu, on applique cette méthode : cela garantit en moyenne une répartition proportionnelle aux poids.
* **Réinitialisation périodique (`_reset_counters`)** : Pour éviter qu’un chemin, une fois saturé par des anciens flux, ne soit sans cesse délaissé, on remet tous les compteurs à 0 toutes les `reset_interval` secondes (10 s dans l’exemple). Cela crée une « fenêtre glissante » d’équilibrage.
* **Logs** : Chaque fois qu’un chemin est sélectionné, la ligne `--> Path selected: port=X` est écrite dans `logs/sdwan_controller.log` avec l’horodatage automatique (` YYYY-MM-DD HH:MM:SS.mmmmmm`). Ces logs seront exploités par le script Mininet.

---

### 2. Script Mininet de simulation : `sdwan_mininet_simulation.py`

```python
#!/usr/bin/env python3
###############################################
# File: sdwan_mininet_simulation.py
# Simulation SD-WAN avec Mininet + génération de trafic + tracé de graphiques
###############################################

import os
import time
import logging
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Répertoire pour stocker les résultats et grâphiques
RESULTS_DIR = 'results'
GRAPH_DIR = os.path.join(RESULTS_DIR, 'graphs')
LOG_DIR = 'logs'  # idem dossier de l’app Ryu

# Paramètres de simulation
SIM_DURATION = 20  # durée (en secondes) du trafic UDP
BW_WAN_1 = 10      # bande passante WAN1 (Mbps)
BW_WAN_2 = 10      # bande passante WAN2 (Mbps)
DELAY_WAN_1 = '10ms'  # latence WAN1
DELAY_WAN_2 = '50ms'  # latence WAN2 (plus élevée)
LOSS_WAN_1 = 0       # perte en % WAN1
LOSS_WAN_2 = 1       # perte en % WAN2

class SDWANTopo(Topo):
    """
    Topologie Mininet :
    - h1 ↔ s1 (succursale A)
    - h2 ↔ s2 (succursale B)
    - Deux chemins WAN en parallèle :
      * s1 ↔ s3 ↔ s2 (WAN1) avec TCLink(BW_WAN_1, DELAY_WAN_1, LOSS_WAN_1)
      * s1 ↔ s4 ↔ s2 (WAN2) avec TCLink(BW_WAN_2, DELAY_WAN_2, LOSS_WAN_2)
    """
    def build(self):
        # Création des switches
        s1 = self.addSwitch('s1')  # CPE succursale A
        s2 = self.addSwitch('s2')  # CPE succursale B
        s3 = self.addSwitch('s3')  # Chemin WAN1
        s4 = self.addSwitch('s4')  # Chemin WAN2

        # Création des hôtes
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')

        # Liens hôtes ↔ CPE
        self.addLink(h1, s1)
        self.addLink(h2, s2)

        # Liaisons CPE ↔ chemins WAN 1 et 2 (TCLink pour simuler bw, delay, loss)
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
    Crée les dossiers results/graphs et logs/ si requise
    """
    for d in [RESULTS_DIR, GRAPH_DIR, LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

def launch_mininet():
    """
    Lance Mininet avec notre topologie SD-WAN et un contrôleur Ryu externe
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
    Génère du trafic UDP entre h1 et h2 pendant SIM_DURATION secondes.
    - h2 écoute sur iperf en UDP
    - h1 envoie 5 Mbps de trafic UDP
    """
    h1 = net.get('h1')
    h2 = net.get('h2')

    # Démarrage du serveur iperf en arrière-plan sur h2
    h2.cmd('iperf -s -u -i 1 > {}/iperf_server.log &'.format(LOG_DIR))
    time.sleep(2)  # laisser le serveur se lancer

    # Lancement du client iperf UDP sur h1
    h1.cmd('iperf -c 10.0.0.2 -u -b 5M -t {} -i 1 > {}/iperf_client.log'.format(SIM_DURATION, LOG_DIR))

def parse_logs_and_plot():
    """
    - Lit logs/sdwan_controller.log pour extraire chaque sélection : "Path selected: port=X"
    - Convertit l’horodatage en secondes relatives
    - Produit :
        * histogramme du nombre de sélections par port
        * courbe de l’évolution temporelle des sélections
    - Sauvegarde les images dans results/graphs/
    """
    controller_log = os.path.join(LOG_DIR, 'sdwan_controller.log')
    if not os.path.isfile(controller_log):
        print(f"Fichier de log non trouvé : {controller_log}")
        return

    ports = []
    timestamps = []
    with open(controller_log, 'r') as f:
        for line in f:
            if 'Path selected: port=' in line or '--> Path selected: port=' in line:
                try:
                    # Exemple de ligne : "2025-06-02 14:23:45.123456 --> Path selected: port=2"
                    parts = line.strip().split()
                    # parts[1] = HH:MM:SS.mmmmmm
                    ts_str = parts[1]
                    port = int(line.strip().split('port=')[1])
                    timestamps.append(ts_str)
                    ports.append(port)
                except:
                    continue

    if not ports:
        print("Aucune sélection de chemin trouvée dans le log.")
        return

    # Conversion des timestamps en secondes relatives
    import datetime
    fmt = '%H:%M:%S.%f'
    t0 = datetime.datetime.strptime(timestamps[0], fmt)
    times_sec = []
    for ts in timestamps:
        t = datetime.datetime.strptime(ts, fmt)
        delta = (t - t0).total_seconds()
        times_sec.append(delta)

    data = {'port': np.array(ports), 'time': np.array(times_sec)}

    # 1) Histogramme du nombre de sélections par port
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

    # 2) Courbe temporelle des sélections de port
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

    print("1) Lancez d’abord Ryu :")
    print("   ryu-manager ryu_sdwan_controller.py")
    print("   (attendez qu’il soit prêt)")
    print("\n2) Ensuite, lancez ce script Mininet :")
    print("   sudo python3 sdwan_mininet_simulation.py")

    net = launch_mininet()
    print("Mininet démarré. Attendez...")
    time.sleep(5)  # laisser le temps aux switches de se connecter au controller

    print(f"Génération de trafic UDP pendant {SIM_DURATION} secondes...")
    run_traffic(net)

    print("Fin du trafic. Arrêt de Mininet...")
    time.sleep(2)
    net.stop()

    print("Analyse des logs et génération des graphiques...")
    parse_logs_and_plot()

if __name__ == '__main__':
    main()
```

#### Détails et points clés

1. **Topologie**

   * **h1 <--> s1**
   * **h2 <--> s2**
   * Deux chemins WAN en parallèle (via s3 et s4) avec paramètres `TCLink` :

     * **WAN1** : bande passante `BW_WAN_1`, latence `DELAY_WAN_1`, perte `LOSS_WAN_1`.
     * **WAN2** : bande passante `BW_WAN_2`, latence `DELAY_WAN_2`, perte `LOSS_WAN_2`.
   * s1 et s2 sont les CPE locaux ; s3 représente le chemin WAN1, s4 le chemin WAN2.

2. **Contrôleur distant**

   * Mininet se connecte à un contrôleur Ryu en local (`127.0.0.1:6633`).
   * Vérifier que `ryu-manager ryu_sdwan_controller.py` a été lancé auparavant.

3. **Génération de trafic**

   * Sur h2 : `iperf -s -u -i 1 > logs/iperf_server.log &` pour lancer le serveur UDP.
   * Sur h1 : `iperf -c 10.0.0.2 -u -b 5M -t SIM_DURATION -i 1 > logs/iperf_client.log`.
   * L’idée est d’envoyer continuellement 5 Mbps en UDP depuis h1 vers h2, ce qui pousse le contrôleur à répartir le trafic entre WAN1 et WAN2.

4. **Analyse des logs et graphiques**

   * On lit `logs/sdwan_controller.log` pour récupérer chaque ligne d’info type `Path selected: port=X`.
   * Conversion de l’horodatage (HH\:MM\:SS.ffffff) en secondes relatives (depuis la première sélection).
   * **Histogramme** : affiche le nombre total de sélections par port (ex. : si WAN2 a un poids deux fois supérieur, on doit constater énviron deux fois plus de sélections sur le port 2).
   * **Évolution temporelle** : pour chaque point, on met l’instant (en s) et le port choisi. On voit ainsi comment au fil du temps le contrôleur alterne entre WAN1 et WAN2.

5. **Organisation des dossiers**

   * `ensure_dirs()` s’assure que

     * `results/graphs/` existe pour y mettre les PNG finaux.
     * `logs/` existe (pour le contrôleur et iperf).
   * Les logs du contrôleur (openFlow) et d’iperf se mélangent dans `logs/`, puis
   * `parse_logs_and_plot()` lit uniquement `sdwan_controller.log` pour les décisions.

---

## Instructions d’utilisation détaillées

1. **Cloner ou créer le répertoire du projet :**

   ```bash
   mkdir projet_sdwan
   cd projet_sdwan
   # Copier/coller les deux fichiers Python et ce README.md ici
   ```

2. **Installer les prérequis (si ce n’est pas encore fait) :**

   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-numpy python3-matplotlib
   sudo pip3 install ryu
   # Installer Mininet (si pas encore en place)
   git clone https://github.com/mininet/mininet.git
   mininet/util/install.sh -a
   ```

3. **Vérifier que Ryu et Mininet fonctionnent :**

   ```bash
   ryu-manager --version   # doit afficher la version Ryu installée
   sudo mn --test pingall   # test Mininet simple
   ```

4. **Lancement du contrôleur Ryu SD-WAN :**
   Dans un terminal (non root), depuis le dossier `projet_sdwan` :

   ```bash
   ryu-manager ryu_sdwan_controller.py
   ```

   * Attendre que Ryu affiche dans la console quelque chose comme :

     ```
     INFO:ryu.base.app_manager:loading app ryu_sdwan_controller
     INFO:ryu.base.app_manager:instantiated app ryu_sdwan_controller
     INFO:ryu.controller.controller:starting ...
     ```
   * À ce stade, le contrôleur est à l’écoute du port 6633 par défaut.

5. **Lancement de la simulation Mininet :**
   Dans un nouveau terminal (root nécessaire pour Mininet) :

   ```bash
   cd projet_sdwan
   sudo python3 sdwan_mininet_simulation.py
   ```

   * Le script va automatiquement :

     1. Créer la topologie (4 switches + 2 hôtes).
     2. Attendre \~5 s pour que tous les switches s’enregistrent auprès de Ryu.
     3. Lancer le serveur `iperf` sur h2, puis envoyer 5 Mbps de trafic UDP depuis h1 pendant `SIM_DURATION` s.
     4. Arrêter Mininet.
     5. Analyser `logs/sdwan_controller.log` pour extraire les sélections de port.
     6. Générer 2 fichiers PNG dans `results/graphs/` :

        * `histogramme_ports.png`
        * `evolution_ports.png`

6. **Résultats attendus :**

   * **Histogramme** : si `PATH_WEIGHTS = {1:1, 2:2}`, on doit voir approximativement deux fois plus de sélections sur le port 2 que sur le port 1.
   * **Courbe temporelle** : des points pour chaque seconde (ou milliseconde si flux plus nombreux) montrant l’alternance.
   * **Logs** : `logs/sdwan_controller.log` contient toutes les sélections, et on peut consulter `logs/iperf_*.log` pour d’autres métriques (latence, perte, etc.).

7. **Personnalisation et optimisation**

   * **Modifier les poids** : dans `ryu_sdwan_controller.py`, changez `PATH_WEIGHTS = {1:1, 2:2}` pour d’autres valeurs, relancez Ryu, puis relancez la simulation.
   * **Changer la latence ou la perte** : dans `sdwan_mininet_simulation.py`, adaptez `DELAY_WAN_1`, `DELAY_WAN_2`, `LOSS_WAN_1` ou `LOSS_WAN_2` pour modéliser un lien non fiable ou rapide.
   * **Ajuster le débit ou la durée UDP** : modifiez `5M` et `SIM_DURATION` dans `run_traffic()`.
   * **Ajouter un 3e chemin WAN** :

     1. Dans `ryu_sdwan_controller.py`, ajoutez une entrée dans `PATH_WEIGHTS`, par exemple `{1:1, 2:2, 3:1}`.
     2. Dans `sdwan_mininet_simulation.py`, ajoutez un switch s5 pour le WAN3, deux liens `s1-s5` et `s5-s2` avec un `TCLink(bw, delay, loss)`.
     3. Lancez Ryu, relancez Mininet.

8. **Extensions possibles**

   * **Classification du trafic** : ajouter du matching sur le protocole (`ipv4`, `tcp`, `udp`) pour donner préférence à un service (par ex. VoIP sur un lien à faible latence).
   * **Monitoring en temps réel** : créer un thread dans Ryu pour relever la latence réelle entre s1 et s2 sur chaque chemin, puis ajuster dynamiquement les poids.
   * **Interactions avec un CPE virtuel** : déployer un hôte, installer un service (serveur web, VPN) pour tester de vraies charges et observer la QoS.

---

## Explications complémentaires et détail des choix

1. **Pourquoi un contrôleur SDN pour SD-WAN ?**

   * Utiliser un contrôleur OpenFlow comme Ryu facilite l’expérimentation. On peut programmer en Python la logique d’équilibrage sans toucher à la conf manuelle des switches.
   * Le contrôleur connait globalement la topologie (via `EventOFPSwitchFeatures`) et peut mesurer ou estimer l’état des liens (par exemple, en combinant les métriques OpenFlow avec du ping).
   * Dans un vrai SD-WAN, on aurait un orchestrateur central capable de communiquer avec des devices physiques (Edge routers, appliances SD-WAN). Ici, on simule simplement avec Mininet.

2. **Algorithme Weighted Round-Robin**

   * Le principe du WRR est simple à implémenter et assez efficace quand la charge est continue.
   * Lorsqu’on veut respecter une proportion de bande passante, on attribue à chaque port un poids et on distribue le trafic en conséquence.
   * Le calcul `counter / weight` choisit le port qui a été le moins servi par rapport à son poids, ce qui évite de « dépenser » tout le trafic sur un seul lien.
   * On peut améliorer : retenir un compteur global de paquets, ou faire un calcul basé sur la consommation réelle de bande passante (via statistiques OpenFlow). Cette version reste basique, mais pédagogique.

3. **Simulation de latence et de perte sur Mininet**

   * Avec `TCLink`, on peut facilement paramétrer :

     * `bw` (bandwidth) en Mbps
     * `delay` (latency) en ms (ex. `10ms`, `50ms`)
     * `loss` (pourcentage de paquets perdus) en %
   * Cette simulation hyper-réaliste permet de voir comment l’algorithme réagit si un lien est plus lent ou sujet à la perte.

4. **Récupération et traitement des logs**

   * Les logs de Ryu sont en format texte, horodatés au format `%Y-%m-%d %H:%M:%S.%f`.
   * On recherche les lignes contenant la chaîne `Path selected: port=X`.
   * On extrait l’heure relative (secondes depuis la 1re sélection) pour tracer la timeline.
   * On peut très bien étendre le script pour exporter en CSV (ex. `timestamp,port`) ou en JSON pour analyses ultérieures.

5. **Production des graphiques**

   * **Matplotlib** a été retenu car :

     * Bibliothèque standard, légère et souvent déjà installée.
     * Pas besoin de frameworks lourds (comme seaborn) pour des graphiques basiques (histogrammes, scatter).
   * Les deux graphiques finaux aident à :

     1. Vérifier qu’on a bien respecté le ratio attendu du WRR.
     2. Visualiser la dynamique de sélection de chemin dans le temps (points alignés ou dispersés ? stable ou oscillant ?).

---

## Vérification des résultats et points de contrôle

1. **Vérifier les logs du contrôleur (`logs/sdwan_controller.log`)**

   * Chaque ligne devrait correspondre à un flux (nouveau paquet) non appris localement.
   * Par exemple :

     ```
     2025-06-02 14:23:45.123456 --> Path selected: port=2
     2025-06-02 14:23:45.234567 --> Path selected: port=1
     ```
   * Le nombre total de lignes `Path selected: port=1` vs `port=2` doit être proportionnel à 1:2 si les poids sont `1` et `2`.

2. **Vérifier les logs `iperf`**

   * Sur h2 (`logs/iperf_server.log`), vous verrez une sortie comme :

     ```

     [ ID] Interval       Transfer     Bandwidth
     [  3]  0.0- 1.0 sec  0.62 MBytes  5.20 Mbits/sec
     [  3]  1.0- 2.0 sec  0.62 MBytes  5.20 Mbits/sec
     ...
     ```
   * Vous pouvez vérifier la qualité du lien (pertes, latence) en croisant avec les paramètres TCLink (ex. WAN2 perd 1%).

3. **Examiner les graphiques**

   * **Histogramme (`histogramme_ports.png`)** :

     * Nombre total de sélections par port. Si `PATH_WEIGHTS = {1:1, 2:2}`, vous devriez observer environ deux fois plus de choix pour le port 2.
   * **Évolution temporelle (`evolution_ports.png`)** :

     * Chaque point = moment où le contrôleur a choisi un port.
     * L’axe Y est simplement le numéro de port (1 ou 2).
     * Vous verrez une alternance comme :

       ```
       Temps (s)   Port
       0.01        2
       0.02        1
       0.03        2
       0.04        2
       0.05        1
       ...  
       ```
     * Cela reflète le WRR : deux fois plus de 2 que de 1 sur une fenêtre de temps.

4. **Tester différents scénarios**

   * **Poids équivalents** (`PATH_WEIGHTS = {1:1, 2:1}`) : on doit obtenir une répartition à peu près 50/50.
   * **Latence différente** (`DELAY_WAN_1 = '10ms'`, `DELAY_WAN_2 = '100ms'`) : le contrôle ne prend pas la latence en compte (dans cette version), mais vous pouvez observer que, même si WAN2 est plus lent, le trafic y est toujours acheminé selon le poids.
   * **Ajouter perte** : augmenter `LOSS_WAN_2` à 5% et constater l’impact sur `iperf`, le contrôleur continuera malgré tout à envoyer les flux sur WAN2 dépendant de ses poids.

---

## Conclusion et perspectives

* La simulation présente une base d’implémentation d’équilibrage de charge dans un contexte SD-WAN. Elle démontre :

  1. Comment modéliser plusieurs liens WAN avec Mininet et `TCLink`.
  2. Comment programmer un contrôleur Ryu pour déployer un algorithme de Weighted Round-Robin.
  3. Comment générer du trafic et analyser les choix de chemin avec des logs et des graphiques.

* **Limites de la version actuelle**

  * Ne prend pas en compte la latence ou la perte actuelles du lien dans le calcul des poids (seulement statiques).
  * Considère qu’un nouveau `PacketIn` = nouveau flux, mais en pratique on devrait agréger par microflows ou utiliser un timeout.
  * S’appuie sur `iperf` en UDP statique (5 Mbps) ; un cas plus réaliste pourrait utiliser des règles QoS OpenFlow ou un agent interne au contrôleur pour suivre la consommation de bande passante.

* **Pistes d’amélioration**

  1. **Collecte de métriques en temps réel** : interroger Ryu ou installer un agent dans Mininet pour mesurer réellement la latence (via ping) ou la bande passante (via `OfStatsRequest`).
  2. **Algorithme adaptatif** : combiner WRR avec un calcul dynamique (ex. pondération basée sur la latence mesurée : si latence > seuil, on diminue temporairement le poids du lien).
  3. **Classification de trafic** : utiliser `OFPMatch` sur IP/port/protocole pour chaîner différents flows dans des files prioritaires, et appliquer un schéma différent pour le trafic critique (VoIP, streaming).
  4. **Interface graphique** : développer un petit tableau de bord pour visualiser en temps réel l’occupation des liens WAN (graphes live).
  5. **Extension multi-sites** : répliquer la topologie pour 3-4 succursales, et faire du trafic entre chaque paire, en déployant un contrôleur central plus périphérique.

---

*Ce README.md contient tous les éléments de contexte, d’explications techniques et les instructions d’usage complètes pour reproduire la simulation d’équilibrage de charge SD-WAN via Mininet et Ryu.*
