"""Plays Grid RTS Game"""
# Some naming conventions used throughout the codebase:
#
# pos vs. cell
# pos (x,y) -- references a point/pixel in windowspace
#               (relative to the topleft corner of the window, or otherwise);
#           ; NEVER references the position of a cell within a grid
# cell (x,y) -- references a cell within a grid; (0,0) is the top left cell, (1,0)
#               is the one directly right of it, etc.
#
# As often as possible, I try to stick to the following naming convention:
# functions/methods and variables (member and non-member) are all lowercase,
# words separated by underscore: get_position(), do(), ...
#
# classes/types are uppercased words, unseparated: CreatureCard, Card, ...
# capitization exceptions: acronyms/initialisms will be all uppercase

import sys
import os
import socket
import selectors
import types
import pygame as pg
import UI
import constants as c
import debug as d
import draw
from input import Input
from game_state import MainMenu, ConnectMenu, HostMenu, Field
from card import BuildingCard, BuilderCard, CreatureCard
from font import fonts
from card_actions import ScanAction

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

class CardPool:
    """Keeps copies of the 'baseline' cards with no modifications"""
    def __init__(self):
        self.cards = []

    def add_card(self, new_card):
        """If the card with the given name is not in the pool, adds it to the pool"""
        for card in self.cards:
            if card.name == new_card.name:
                print("Tried to add card to card pool with same name as one already in pool. Card not added.")
                return {'success': False}

        self.cards.append(new_card)
        return {'success': True}

    def get_card_by_name(self, name):
        """If the card with the given name is in the pool, returns a clone of it."""
        for card in self.cards:
            if card.name == name:
                return card.clone()

        return None

class Game:
    """Delegates local input, frame-by-frame updates, and network input to objects in the game"""
    def __init__(self, screen, start_state=None, debug=False):
        if debug is True:
            d.DebugUI(game=self, active=True)

        self.screen = screen

        self.card_pool = CardPool()

        goblin_card_prototype = CreatureCard(name='Goblin', cost=2, base_power=1, max_health=2, visibility=[1])
        elf_card_prototype = CreatureCard(name='Elf', cost=1, base_power=1, max_health=1, visibility=[1,0,0,0])
        scout_card_prototype = CreatureCard(name='Scout', cost=1, base_power=0, max_health=1, visibility=[3,0,0,0])
        drone_card_prototype = BuilderCard(name='Drone', cost=1, base_power=0, max_health=1, visibility=[1])
        blacksmith_card_prototype = BuildingCard(name='Blacksmith', cost=1, max_health=4, visibility=[0])
        satellite_card_prototype = BuildingCard(name='Satellite', cost=1, max_health=3, visibility=[0],
                                                default_action='SCN',
                                                active_actions={'SCN': lambda card: ScanAction})
        tower_card_prototype = BuildingCard(name='Tower', cost=2, max_health=4, visibility=[3,0,0,0])

        self.card_pool.add_card(goblin_card_prototype)
        self.card_pool.add_card(elf_card_prototype)
        self.card_pool.add_card(scout_card_prototype)
        self.card_pool.add_card(drone_card_prototype)
        self.card_pool.add_card(blacksmith_card_prototype)
        self.card_pool.add_card(satellite_card_prototype)
        self.card_pool.add_card(tower_card_prototype)

        self.network_data_queue = []

        self.connection_label = UI.Label(   pos=(0,0),
                                            font=fonts.main_menu_font_small,
                                            text='',
                                            text_color=c.green,
                                            align=('left','up'))

        self.chat_window = UI.ChatWindow(   name_font=fonts.chat_name_font,
                                            message_font=fonts.chat_message_font,
                                            name_width=100,
                                            message_width=300,
                                            log_height=150)

        self.ui_group = None
        self.ui_container = UI.Container()
        self.ui_container.add_element(self.connection_label)
        self.ui_container.add_element(self.chat_window)

        self.input_map = {
            Input(key='any'): lambda key, mod, unicode_key: self.any_key_pressed(key=key, mod=mod, unicode_key=unicode_key),
            Input(mouse_button=1): lambda mouse_pos: self.left_mouse_pressed(mouse_pos),
            Input(mouse_button=1, type='release'): lambda mouse_pos: self.left_mouse_released(mouse_pos)
        }

        self._state = None
        self.state = start_state(self)

        self.selector = None
        self.socket = None
        self.connected_to_address = None
        self.connection_role = None
        self._connected = False
        self._accepting_connections = False

    def refresh_ui_group(self):
        """Set the UI group to contain only the game UI container and its current state's UI container"""
        self.ui_group = UI.Group(containers=[self.ui_container, self.state.ui_container], screen=self.screen)

    def any_key_pressed(self, key, mod, unicode_key):
        """Pass input event down through UI and game elements"""
        self.ui_group.any_key_pressed(key, mod, unicode_key)

    def left_mouse_pressed(self, mouse_pos):
        """Pass input event down through UI and game elements"""
        self.ui_group.left_mouse_pressed(mouse_pos)

    def left_mouse_released(self, mouse_pos):
        """Pass input event down through UI and game elements"""
        self.ui_group.left_mouse_released(mouse_pos)

    def add_chat_message(self, user, text):
        """Add message to chat window"""
        self.chat_window.add_message(user=user, text=text)

    @property
    def state(self):
        """Getter"""
        return self._state

    @state.setter
    def state(self, new_state):
        """Whenever the state changes, update the UI group to remove the old state's UI container,
        and add the new state's UI container. Then call the new state's enter() method."""
        self._state = new_state
        self.refresh_ui_group()
        self._state.enter()

    @property
    def connected(self):
        """Getter"""
        return self._connected

    @connected.setter
    def connected(self, new_state):
        """Sync the UI to show whether the client app is currently connected to a host"""
        if new_state is True:
            self.connection_label.text = "Connected to %s"%str(self.connected_to_address)
        else:
            self.connection_label.text = ''

        self._connected = new_state

    @property
    def accepting_connections(self):
        """Getter"""
        return self._accepting_connections

    @accepting_connections.setter
    def accepting_connections(self, new_state):
        """Sync the UI to show whether the host app is currently accepting connections"""
        if new_state is True:
            self.connection_label.text = "Accepting Connections"
        else:
            self.connection_label.text = ''

        self._accepting_connections = new_state


    def start_host(self, port):
        """Open the host for incoming connections"""
        self.selector = selectors.DefaultSelector()
        host = '0.0.0.0'

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host,port))
        self.socket.listen()
        self.socket.setblocking(False)
        print("Now accepting connections %s:%s"%(host, port))
        self.selector.register(self.socket, selectors.EVENT_READ, data=None)
        self.accepting_connections = True

    def _attempt_to_accept_connection(self, sock):
        """As host, attempt to accept incoming connection request"""
        connection, self.connected_to_address = sock.accept()
        print('Accepted connection from' , self.connected_to_address)
        self.connection_label.text = "Connected to %s"%str(self.connected_to_address)
        connection.setblocking(False)
        data = types.SimpleNamespace(addr=self.connected_to_address, inb=b"", outb=b"")
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        self.selector.register(connection, events, data=data)
        self.accepting_connections = False
        self.connected = True
        self.connection_role = 'host'

    def _attempt_to_connect(self, host, port):
        """As client, attempt to connect to the 'host' through 'port'"""
        self.selector = selectors.DefaultSelector()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setblocking(False)
        self.socket.connect_ex((host,port))
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        data = types.SimpleNamespace(   connid=0,
                                        msg_total=1,
                                        recv_total=0,
                                        messages=[b'SIMON IS CUTE'],
                                        outb=b"")

        self.selector.register(self.socket, events, data=data)
        self.connection_role = 'client'
        print('Connected to %s:%s'%(host, port))
        self.connected = True
        self.connection_label.text = 'Connected to %s:%s'%(host, port)

    def _service_connection_as_host(self, key, mask):
        """When this instance is the host, send and receive packets"""
        sock = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)
            if recv_data:
                #data.outb += recv_data
                print('received', repr(recv_data))
                self.process_network_data(recv_data)
        elif mask & selectors.EVENT_WRITE:
            for packet in self.network_data_queue:
                print('sending', repr(packet), 'to', data.addr)
                sock.send(packet)

            self.network_data_queue = []

    def _service_connection_as_client(self, key, mask):
        """When this instance is the client, send and receive packets"""
        sock = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)  # Should be ready to read
            if recv_data:
                print("received", repr(recv_data), "from connection", data.connid)
                self.process_network_data(recv_data)
        elif mask & selectors.EVENT_WRITE:
            for packet in self.network_data_queue:
                print("sending", repr(packet), "to connection", data.connid)
                sock.send(packet)  # Should be ready to write

            self.network_data_queue = []

    def close_connection(self):
        """Close connection and clear all state related to it"""
        if self.accepting_connections or self.connected:
            print('Closing connection')
            self.selector.unregister(self.socket)
            self.socket.close()

            self.selector = None
            self.socket = None
            self.connected = False
            self.accepting_connections = False
            self.connection_role = None
        else:
            print('Connection already closed')


    def select(self):
        """Look for new packet arrivals"""
        if self.selector:
            events = self.selector.select(timeout=0)
            for key, mask in events:
                if key.data is None:
                    self._attempt_to_accept_connection(key.fileobj)
                else:
                    if self.connection_role == 'host':
                        self._service_connection_as_host(key, mask)
                    elif self.connection_role == 'client':
                        self._service_connection_as_client(key, mask)

    def queue_network_data(self, data):
        """Queue argument 'data' as a packet, to be processed later"""
        self.network_data_queue.append(data)

    def process_network_data(self, data):
        """Process argument 'data' as a packet, and change state accordingly.
        The packet is passed down to the game state as well for further processing"""
        print('process_network_data')
        raw_data_string = data.decode('utf-8')
        event_strings = raw_data_string.split('[END]')
        for event_string in event_strings:
            args = event_string.split(';')
            try:
                if args[0] == 'message sent':
                    self.chat_window.add_message(args[1], args[2])
            except IndexError:
                print("Invalid args received in network packet.")

        self.state.process_network_data(data)

    # def is_valid_player(self, player):
    #     if player == 0 or player == 1:
    #         return True
    #     else:
    #         return False

    @property
    def board(self):
        """Returns the board if the game state is currently in a game"""
        if not isinstance(self.state, Field):
            return None

        return self.state.board

    def handle_input(self, input_data, mouse_pos, mod=None, unicode_key=None):
        """Read input and make any state changes based on that input"""
        if input_data in self.input_map:
            self.input_map[input_data](mouse_pos)
        else:
            self.input_map[Input(key='any')](input_data.key, mod, unicode_key)

        if self.ui_group.focused_container is None or self.ui_group.focused_container.focused_element is None:
            self.state.handle_input(input_data, mouse_pos, mod, unicode_key)

    def get_player_name(self):
        """Temporary way to assign host and client distinct names in chat"""
        if self.connection_role == 'host':
            return 'Tyler'
        if self.connection_role == 'client':
            return 'Shawn'

        return 'OFFLINE'

    def update(self, dt, mouse_pos):
        """Do all frame-by-frame updates"""
        if self.state.target_state:
            self.state = self.state.target_state

        self.select()
        self.state.update(dt, mouse_pos)
        self.ui_group.update(dt, mouse_pos)

        for ui_element in self.ui_group:
            event = ui_element.get_event()
            while event is not None:
                if event[0] == 'send chat message':
                    self.chat_window.add_message(user=self.get_player_name(), text=event[1])
                    send_string = 'message sent;' + self.get_player_name() + ';' + event[1] + ';[END]'
                    self.queue_network_data(send_string.encode('utf-8'))

                event = ui_element.get_event()


        # if self.ui_group.focused_container:
        #   if self.ui_group.focused_container.focused_element:
        #       print(self.ui_group.focused_container.focused_element)
        # else:
        #   print('**')

    def draw(self):
        "Draw everything related to the game"
        self.state.draw()
        self.ui_group.draw()

def parse_window_settings():
    """Parse window settings given by cmd line; used for placing the window somewhere,
    for easier multi-window debugging
    """
    for arg in sys.argv:
        if arg == '-left':
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (10,30)
        elif arg == '-right':
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (700,30)

parse_window_settings()

# pg setup
pg.init()
fonts.load()
pg.key.set_repeat(300, 30)
draw.screen = pg.display.set_mode(c.screen_size)

# Game setup
game_clock = pg.time.Clock()

#input = Input()

def parse_game_settings():
    """Parse game settings given by cmd line; mostly used for quicker debugging"""
    start_state = lambda game_: MainMenu(game_)
    debug = False
    for arg in sys.argv:
        if arg == '-field':
            start_state = lambda game_: Field(game_, 0, 'SP')
        elif arg == '-connect':
            start_state = lambda game_: ConnectMenu(game_)
        elif arg == '-host':
            start_state = lambda game_: HostMenu(game_)
        elif arg == '-d':
            debug = True

    return Game(screen=draw.screen, start_state=start_state, debug=debug)

game = parse_game_settings()

def handle_events():
    """""Handle window events: input, quit, etc."""
    for event in pg.event.get():
        if event.type == pg.QUIT:
            sys.exit()
        elif event.type == pg.MOUSEBUTTONDOWN:
            input_data = Input(mouse_button=event.button, type='press')
            game.handle_input(input_data=input_data, mouse_pos=event.pos)
        elif event.type == pg.MOUSEBUTTONUP:
            input_data = Input(mouse_button=event.button, type='release')
            game.handle_input(input_data=input_data, mouse_pos=event.pos)
        elif event.type == pg.KEYDOWN:
            input_data = Input(key=event.key, type='press')
            game.handle_input(input_data=input_data, mouse_pos=pg.mouse.get_pos(), mod=event.mod, unicode_key=event.unicode)

while True:
    # Event loop
    handle_events()

    # Update
    game.update(dt=game_clock.tick(60), mouse_pos=pg.mouse.get_pos())

    # Draw
    draw.screen.fill(c.black)
    game.draw()

    # Push screen
    pg.display.flip()
