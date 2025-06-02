#!/usr/bin/env python3
###############################################
# File: ryu_sdwan_controller_fixed.py
# Contrôleur Ryu SD-WAN avec logs compatibles simulation
###############################################

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
from ryu.lib import hub
import logging
import time
import os
import json
from datetime import datetime

# Configuration des chemins WAN avec critères multiples
WAN_PATHS = {
    1: {'weight': 3, 'name': 'MPLS', 'priority': 'high'},
    2: {'weight': 2, 'name': 'Fiber', 'priority': 'medium'}, 
    3: {'weight': 1, 'name': '4G', 'priority': 'low'}
}

# Répertoires
LOG_DIR = 'logs'
STATS_DIR = 'stats'
LOG_FILE = os.path.join(LOG_DIR, 'sdwan_controller.log')
STATS_FILE = os.path.join(STATS_DIR, 'path_statistics.json')

class SDWANController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SDWANController, self).__init__(*args, **kwargs)
        
        # Création des dossiers
        for d in [LOG_DIR, STATS_DIR]:
            if not os.path.exists(d):
                os.makedirs(d)
        
        # ✅ CORRECTION 1: Configuration logging avec format timestamp simple
        # Supprimer le logging existant et reconfigurer
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            filename=LOG_FILE,
            format='%(asctime)s %(message)s',
            datefmt='%H:%M:%S.%f',
            level=logging.INFO,
            filemode='w'
        )
        
        # ✅ AJOUT: Logger console pour debug
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S.%f'))
        logging.getLogger().addHandler(console_handler)
        
        # Structures de données
        self.mac_to_port = {}
        self.flow_stats = {}
        self.path_counters = {}
        self.path_performance = {}
        
        # Initialisation des compteurs WAN
        for path_id, config in WAN_PATHS.items():
            self.path_counters[path_id] = {
                'weight': config['weight'],
                'counter': 0,
                'bytes_sent': 0,
                'packets_sent': 0,
                'name': config['name'],
                'priority': config['priority']
            }
            
        # Statistiques temporelles
        self.stats_history = []
        self.reset_interval = 15
        self.stats_thread = hub.spawn(self._periodic_stats_collection)
        
        # ✅ CORRECTION 2: Logs d'initialisation au bon format
        logging.info("=== SDWAN Controller Initialized ===")
        logging.info(f"WAN Paths configured: {len(WAN_PATHS)}")
        for path_id, config in WAN_PATHS.items():
            logging.info(f"Path {path_id}: {config['name']} (weight: {config['weight']}, priority: {config['priority']})")
            
        print(f"[CONTROLLER] Log file created: {LOG_FILE}")

    def _periodic_stats_collection(self):
        """Collecte périodique des statistiques"""
        while True:
            hub.sleep(self.reset_interval)
            
            # Sauvegarde des stats actuelles
            timestamp = datetime.now().isoformat()
            current_stats = {
                'timestamp': timestamp,
                'paths': dict(self.path_counters),
                'total_flows': sum(p['counter'] for p in self.path_counters.values())
            }
            self.stats_history.append(current_stats)
            
            # Log détaillé
            logging.info("=== PERIODIC STATISTICS ===")
            for path_id, stats in self.path_counters.items():
                logging.info(f"Path {path_id} ({stats['name']}): {stats['counter']} flows, {stats['packets_sent']} packets")
            
            # Sauvegarde JSON
            try:
                with open(STATS_FILE, 'w') as f:
                    json.dump(self.stats_history, f, indent=2)
            except Exception as e:
                logging.error(f"Error saving stats: {e}")
            
            # Reset des compteurs pour la prochaine période
            for path_id in self.path_counters:
                self.path_counters[path_id]['counter'] = 0
            
            logging.info("Counters reset - New measurement period started")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Configuration initiale du switch"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        logging.info(f"Switch connected: DPID {datapath.id}")
        print(f"[CONTROLLER] Switch {datapath.id} connected")
        
        # Installation de la table-miss
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0):
        """Ajoute un flow dans le switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(
                datapath=datapath, buffer_id=buffer_id, priority=priority,
                match=match, instructions=inst, idle_timeout=idle_timeout
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath, priority=priority, match=match,
                instructions=inst, idle_timeout=idle_timeout
            )
        datapath.send_msg(mod)

    def select_path_intelligent(self, src_ip=None, dst_ip=None, protocol=None):
        """
        ✅ CORRECTION 3: Sélection intelligente avec logs compatibles
        """
        
        # Algorithme de base : Weighted Round Robin
        best_path = None
        best_ratio = float('inf')
        
        for path_id, data in self.path_counters.items():
            if data['weight'] > 0:  # Éviter division par zéro
                ratio = data['counter'] / data['weight']
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_path = path_id
        
        # Si pas de chemin trouvé, utiliser le premier disponible
        if best_path is None:
            best_path = list(WAN_PATHS.keys())[0]
        
        # Mise à jour des compteurs
        self.path_counters[best_path]['counter'] += 1
        self.path_counters[best_path]['packets_sent'] += 1
        
        # ✅ CORRECTION 4: Log au format attendu par la simulation
        path_name = WAN_PATHS[best_path]['name']
        logging.info(f"Path selected: port={best_path} ({path_name})")
        print(f"[CONTROLLER] --> Path selected: port={best_path} ({path_name})")
        
        return best_path

    def _is_inter_branch_traffic(self, src_ip, dst_ip):
        """
        ✅ CORRECTION 5: Détection du trafic inter-branches
        """
        if not src_ip or not dst_ip:
            return False
            
        # Branch A: 10.1.0.x, Branch B: 10.2.0.x
        src_branch = None
        dst_branch = None
        
        if src_ip.startswith('10.1.0.'):
            src_branch = 'A'
        elif src_ip.startswith('10.2.0.'):
            src_branch = 'B'
            
        if dst_ip.startswith('10.1.0.'):
            dst_branch = 'A'
        elif dst_ip.startswith('10.2.0.'):
            dst_branch = 'B'
            
        return src_branch and dst_branch and src_branch != dst_branch

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """✅ CORRECTION 6: Gestionnaire avec forced WAN routing"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        # Analyse du paquet
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        dst = eth.dst
        src = eth.src
        
        # Informations IP pour décision intelligente
        src_ip = None
        dst_ip = None
        protocol = None
        
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            src_ip = ip_pkt.src
            dst_ip = ip_pkt.dst
            protocol = ip_pkt.proto
        
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        
        # Apprentissage MAC
        self.mac_to_port[dpid][src] = in_port
        
        # ✅ DÉCISION DE ROUTAGE FORCÉE POUR TRAFIC INTER-BRANCHES
        decision_type = "LOCAL"
        out_port = None
        
        # Forcer l'équilibrage pour trafic inter-branches
        if self._is_inter_branch_traffic(src_ip, dst_ip):
            out_port = self.select_path_intelligent(src_ip, dst_ip, protocol)
            decision_type = "WAN_LOAD_BALANCED"
            
            logging.info(f"Inter-branch traffic: {src_ip} -> {dst_ip} via port {out_port}")
            
        # Routage local normal
        elif dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
            decision_type = "LOCAL_LEARNED"
        else:
            # ✅ FALLBACK: Si destination inconnue, forcer équilibrage WAN
            out_port = self.select_path_intelligent(src_ip, dst_ip, protocol)
            decision_type = "WAN_UNKNOWN_DST"
            
            logging.info(f"Unknown destination: {dst} -> routing via WAN port {out_port}")
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Installation du flow avec timeout plus court pour plus d'activité
        if msg.buffer_id != ofproto.OFP_NO_BUFFER:
            match = parser.OFPMatch(eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle_timeout=10)  # Réduit de 30 à 10
            return
        
        match = parser.OFPMatch(eth_dst=dst, eth_src=src)
        self.add_flow(datapath, 1, match, actions, idle_timeout=10)  # Réduit de 30 à 10
        
        # Envoi du paquet
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
            
        out = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data
        )
        datapath.send_msg(out)