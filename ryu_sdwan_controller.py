#!/usr/bin/env python3
###############################################
# File: ryu_sdwan_controller_v2.py
# Contrôleur Ryu SD-WAN adapté pour trafic Internet
###############################################

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, arp
from ryu.lib import hub
import logging
import time
import os
import json
from datetime import datetime

# Configuration des chemins WAN avec critères multiples
WAN_PATHS = {
    1: {'weight': 3, 'name': 'MPLS', 'priority': 'high', 'type': 'premium'},
    2: {'weight': 2, 'name': 'Fiber', 'priority': 'medium', 'type': 'high_bandwidth'}, 
    3: {'weight': 1, 'name': '4G', 'priority': 'low', 'type': 'backup'}
}

# Mapping des types de trafic vers les préférences de liens
TRAFFIC_PREFERENCES = {
    'web': [1, 2, 3],      # MPLS préféré pour web (faible latence)
    'video': [2, 1, 3],    # Fiber préféré pour vidéo (haute BP)
    'voip': [1, 2, 3],     # MPLS obligatoire pour VoIP (faible latence)
    'data': [2, 3, 1],     # Fiber puis 4G pour gros transferts
    'default': [1, 2, 3]   # Par défaut
}

# Répertoires
LOG_DIR = 'logs'
STATS_DIR = 'stats'
LOG_FILE = os.path.join(LOG_DIR, 'sdwan_controller.log')
STATS_FILE = os.path.join(STATS_DIR, 'path_statistics.json')

class SDWANControllerV2(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SDWANControllerV2, self).__init__(*args, **kwargs)
        
        # Création des dossiers
        for d in [LOG_DIR, STATS_DIR]:
            if not os.path.exists(d):
                os.makedirs(d)
        
        # Configuration du logging détaillé
        logging.basicConfig(
            filename=LOG_FILE,
            format='%(asctime)s | %(levelname)s | %(message)s',
            level=logging.INFO,
            filemode='w'
        )
        
        # Structures de données étendues
        self.mac_to_port = {}
        self.ip_to_mac = {}
        self.flow_stats = {}
        self.path_counters = {}
        self.internet_servers = {}
        
        # Initialisation des compteurs WAN
        for path_id, config in WAN_PATHS.items():
            self.path_counters[path_id] = {
                'weight': config['weight'],
                'counter': 0,
                'bytes_sent': 0,
                'packets_sent': 0,
                'name': config['name'],
                'priority': config['priority'],
                'type': config['type']
            }
        
        # Reconnaissance des serveurs Internet (par IP)
        self.internet_ips = {
            '8.8.8.8': 'Google_DNS',
            '1.1.1.1': 'Cloudflare',
            '172.217.1.1': 'YouTube',
            '54.230.1.1': 'Netflix',
            '54.239.1.1': 'AWS',
            '13.107.1.1': 'Azure',
            '192.168.100.1': 'Backup_Server'
        }
        
        # Statistiques temporelles
        self.stats_history = []
        self.reset_interval = 15
        self.stats_thread = hub.spawn(self._periodic_stats_collection)
        
        logging.info("=== SDWAN Controller V2 (Internet Traffic) Initialized ===")
        logging.info(f"WAN Paths configured: {len(WAN_PATHS)}")
        for path_id, config in WAN_PATHS.items():
            logging.info(f"Path {path_id}: {config['name']} (weight: {config['weight']}, type: {config['type']})")
        
        logging.info(f"Internet servers monitored: {len(self.internet_ips)}")
        for ip, name in self.internet_ips.items():
            logging.info(f"Server: {ip} -> {name}")

    def _periodic_stats_collection(self):
        """Collecte périodique des statistiques avec détails Internet"""
        while True:
            hub.sleep(self.reset_interval)
            
            timestamp = datetime.now().isoformat()
            current_stats = {
                'timestamp': timestamp,
                'paths': dict(self.path_counters),
                'total_flows': sum(p['counter'] for p in self.path_counters.values()),
                'internet_traffic': {
                    'servers_accessed': len(self.internet_servers),
                    'total_internet_flows': sum(1 for f in self.flow_stats.values() if f.get('is_internet', False))
                }
            }
            self.stats_history.append(current_stats)
            
            # Log détaillé avec focus Internet
            logging.info("=== PERIODIC STATISTICS (INTERNET TRAFFIC) ===")
            total_internet = sum(1 for f in self.flow_stats.values() if f.get('is_internet', False))
            logging.info(f"Total Internet flows: {total_internet}")
            
            for path_id, stats in self.path_counters.items():
                logging.info(f"Path {path_id} ({stats['name']}): {stats['counter']} flows, {stats['packets_sent']} packets")
            
            # Log des serveurs Internet les plus utilisés
            if self.internet_servers:
                logging.info("Top Internet destinations:")
                sorted_servers = sorted(self.internet_servers.items(), key=lambda x: x[1], reverse=True)
                for server_ip, count in sorted_servers[:3]:
                    server_name = self.internet_ips.get(server_ip, 'Unknown')
                    logging.info(f"  {server_name} ({server_ip}): {count} connections")
            
            # Sauvegarde JSON
            with open(STATS_FILE, 'w') as f:
                json.dump(self.stats_history, f, indent=2)
            
            # Reset des compteurs
            for path_id in self.path_counters:
                self.path_counters[path_id]['counter'] = 0
            
            logging.info("Counters reset - New measurement period started")

    def detect_traffic_type(self, src_ip, dst_ip, protocol, src_port=None, dst_port=None):
        """Détecte le type de trafic basé sur IP de destination et ports"""
        
        # Vérification si c'est du trafic Internet
        is_internet = dst_ip in self.internet_ips
        
        if not is_internet:
            return 'internal', False
        
        # Classification par destination Internet
        server_name = self.internet_ips[dst_ip]
        
        if server_name in ['Google_DNS', 'Cloudflare']:
            return 'web', True
        elif server_name in ['YouTube', 'Netflix']:
            return 'video', True
        elif server_name in ['AWS', 'Azure']:
            return 'cloud', True
        elif server_name == 'Backup_Server':
            return 'data', True
        
        # Classification par port si pas reconnu par IP
        if dst_port:
            if dst_port in [80, 443, 8080]:
                return 'web', True
            elif dst_port in [1234, 5004]:  # Ports vidéo utilisés par iperf
                return 'video', True
            elif dst_port in [22, 21, 2049]:  # SSH, FTP, NFS
                return 'data', True
        
        return 'default', is_internet

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Configuration initiale du switch"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto
        parser = datapath.ofproto_parser
        
        logging.info(f"Switch connected: DPID {datapath.id}")
        
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

    def select_path_intelligent(self, traffic_type='default', dst_ip=None):
        """
        Sélection intelligente de chemin basée sur :
        - Type de trafic (web, video, cloud, data)
        - Weighted Round Robin
        - Préférences par type d'application
        """
        
        # Obtenir les préférences pour ce type de trafic
        preferred_paths = TRAFFIC_PREFERENCES.get(traffic_type, TRAFFIC_PREFERENCES['default'])
        
        # Algorithme WRR sur les chemins préférés
        best_path = None
        best_ratio = float('inf')
        
        for path_id in preferred_paths:
            if path_id in self.path_counters:
                data = self.path_counters[path_id]
                if data['weight'] > 0:
                    ratio = data['counter'] / data['weight']
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_path = path_id
        
        # Fallback si pas de chemin trouvé
        if best_path is None:
            best_path = list(WAN_PATHS.keys())[0]
        
        # Mise à jour des compteurs
        self.path_counters[best_path]['counter'] += 1
        self.path_counters[best_path]['packets_sent'] += 1
        
        # Statistiques serveurs Internet
        if dst_ip and dst_ip in self.internet_ips:
            if dst_ip not in self.internet_servers:
                self.internet_servers[dst_ip] = 0
            self.internet_servers[dst_ip] += 1
        
        return best_path

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Gestionnaire principal des paquets entrants - adapté pour trafic Internet"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        # Analyse du paquet
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        dst_mac = eth.dst
        src_mac = eth.src
        
        # Informations IP pour décision intelligente
        src_ip = None
        dst_ip = None
        protocol = None
        src_port = None
        dst_port = None
        
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            src_ip = ip_pkt.src
            dst_ip = ip_pkt.dst
            protocol = ip_pkt.proto
            
            # Extraction des ports pour TCP/UDP
            tcp_pkt = pkt.get_protocol(tcp.tcp)
            udp_pkt = pkt.get_protocol(udp.udp)
            
            if tcp_pkt:
                src_port = tcp_pkt.src_port
                dst_port = tcp_pkt.dst_port
            elif udp_pkt:
                src_port = udp_pkt.src_port
                dst_port = udp_pkt.dst_port
        
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        
        # Apprentissage MAC
        self.mac_to_port[dpid][src_mac] = in_port
        if src_ip:
            self.ip_to_mac[src_ip] = src_mac
        
        # Classification du trafic
        traffic_type, is_internet = self.detect_traffic_type(src_ip, dst_ip, protocol, src_port, dst_port)
        
        # Décision de routage
        if dst_mac in self.mac_to_port[dpid] and not is_internet:
            # Trafic local connu
            out_port = self.mac_to_port[dpid][dst_mac]
            decision_type = "LOCAL_LEARNED"
        else:
            # Trafic WAN (Internet ou inter-sites)
            out_port = self.select_path_intelligent(traffic_type, dst_ip)
            decision_type = "WAN_INTERNET" if is_internet else "WAN_INTER_SITE"
            
            # Log détaillé pour trafic Internet
            if is_internet:
                server_name = self.internet_ips.get(dst_ip, 'Unknown')
                path_info = WAN_PATHS.get(out_port, {})
                logging.info(f"INTERNET_FLOW | SRC: {src_ip} | DST: {dst_ip} ({server_name}) | "
                           f"TRAFFIC_TYPE: {traffic_type} | PATH: {out_port} ({path_info.get('name', 'Unknown')}) | "
                           f"PROTOCOL: {protocol} | PORT: {dst_port}")
                
                # Enregistrement du flux
                flow_key = f"{src_ip}-{dst_ip}-{src_port}-{dst_port}"
                self.flow_stats[flow_key] = {
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'traffic_type': traffic_type,
                    'path_used': out_port,
                    'is_internet': True,
                    'server_name': server_name,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Log pour trafic non-Internet
                path_info = WAN_PATHS.get(out_port, {})
                logging.info(f"WAN_FLOW | SRC: {src_ip} | DST: {dst_ip} | "
                           f"PATH: {out_port} ({path_info.get('name', 'Unknown')}) | "
                           f"TYPE: {decision_type}")
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Installation du flow avec timeout adapté au type de trafic
        timeout = 60 if is_internet else 30  # Plus long pour trafic Internet
        
        if msg.buffer_id != ofproto.OFP_NO_BUFFER:
            match = parser.OFPMatch(eth_dst=dst_mac, eth_src=src_mac)
            self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle_timeout=timeout)
            return
        
        match = parser.OFPMatch(eth_dst=dst_mac, eth_src=src_mac)
        self.add_flow(datapath, 1, match, actions, idle_timeout=timeout)
        
        # Envoi du paquet
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
            
        out = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data
        )
        datapath.send_msg(out)