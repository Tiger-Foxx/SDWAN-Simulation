###############################################
# File: ryu_sdwan_controller.py
# Ryu application pour l’équilibrage de charge SD-WAN pondéré
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

# --- Configuration des chemins WAN ---
# Les clés du dictionnaire sont les numéros de port du switch (dédiés aux liens WAN).
# Les valeurs sont les poids : plus le poids est élevé, plus ce chemin sera utilisé.
PATH_WEIGHTS = {1: 1, 2: 2}

# Répertoire et fichier de log
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

        # Table MAC → port pour apprentissage
        self.mac_to_port = {}

        # Initialisation des compteurs pour chaque chemin WAN
        self.path_counters = {}
        for port, weight in PATH_WEIGHTS.items():
            self.path_counters[port] = {'weight': weight, 'counter': 0}

        # Thread vert pour réinitialiser périodiquement les compteurs (optionnel)
        self.reset_interval = 10  # en secondes
        self.monitor_thread = hub.spawn(self._reset_counters)

    def _reset_counters(self):
        while True:
            hub.sleep(self.reset_interval)
            for port in self.path_counters:
                self.path_counters[port]['counter'] = 0
            logging.info("‼ Counters reset pour tous les chemins WAN")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        Installation de la table-miss pour rediriger tous les paquets inconnus vers le contrôleur
        """
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Match qui capte tout
        match = parser.OFPMatch()
        # Action : renvoyer au contrôleur
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        # Priorité 0 → table-miss
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """
        Ajoute une entrée de flux (flow) dans le switch.
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
        Sélection pondérée d’un port WAN selon l’algorithme round-robin.
        On calcule pour chaque port le ratio (counter / weight) et on choisit celui de plus petit ratio.
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
        # On incrémente le compteur pour le port sélectionné
        self.path_counters[best_port]['counter'] += 1
        return best_port

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
        Gère les événements PacketIn :
        - Apprentissage MAC → port
        - Si la destination est connue en local, on envoie directement
        - Sinon, on applique notre équilibrage de charge pondéré pour choisir le chemin WAN
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
        # Initialisation du dictionnaire pour ce switch
        self.mac_to_port.setdefault(dpid, {})

        # Apprentissage de la source
        self.mac_to_port[dpid][src] = in_port

        # Si on connaît déjà le port de destination en interne → on forward
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            # Sinon on choisit un chemin WAN selon l’algo pondéré
            out_port = self.select_path()
            logging.info(f"--> Path selected: port={out_port}")

        actions = [parser.OFPActionOutput(out_port)]

        # Installation d’un flow pour éviter des futurs PacketIn
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
