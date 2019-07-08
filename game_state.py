import UI
import sys
from input import Input
from board import Board
import constants as c
from hand import Hand
from phases import Phase, TurnDisplay
from font import fonts
import pygame as pg
import draw
import util
from card import *

import random

class GameState:
	def __init__(self, game):
		self.game = game
		self.ui_container = UI.Container()
		self.target_state = None
	def handle_input(self, input, mouse_pos, mod=None, unicode_key=None):
		if input in self.input_map:
			self.input_map[input](mouse_pos)
		elif Input(key='any') in self.input_map:
			self.input_map[Input(key='any')](input.key, mod, unicode_key)
		# state = None
		# if input in self.input_map:
		# 	state = self.input_map[input](mouse_pos)

		# if state != None:
		# 	return state 	# if we have a state change, go ahead and return,
		# else:				# but if not, let's check the 'any key' event:
		# 	if input.key:
		# 		state = self.input_map[Input(key='any')](input.key, mod, unicode_key)
	def any_key_pressed(self, key, mod, unicode_key):
		pass#self.ui_container.any_key_pressed(key, mod, unicode_key)
	def left_mouse_pressed(self, mouse_pos):
		pass#self.ui_container.left_mouse_pressed(mouse_pos)
	def enter(self):
		pass
	def exit(self):
		pass
	def update(self, dt, mouse_pos):
		pass
	# These are 'virtual' methods -- should be overridden by child class
	def draw(self):
		pass
	def queue_state_transition(self, new_state):
		self.target_state = new_state
	def fetch_network_data(self):
		pass
	def process_network_data(self, data):
		pass

class MainMenu(GameState):
	def __init__(self, game):
		GameState.__init__(self, game)

		self.input_map = {
			#Input(key='any'): lambda key, mod, unicode_key: self.any_key_pressed(key, mod, unicode_key),
			Input(key=pg.K_ESCAPE): lambda _: sys.exit()
		}


		self.list_menu = UI.ListMenu(	items=('Start SP Game', 'Start MP Game', 'Host', 'Connect', 'Exit'),
									pos=(300,300),
									align=('center','center'),
									text_align=('center'),
									font=fonts.main_menu_font,
									selected_font=fonts.main_menu_selected_font)

		self.ui_container.add_element(self.list_menu)

	def enter(self):
		self.ui_container.focus_element(self.list_menu)

	def update(self, dt, mouse_pos):
		if self.list_menu.confirmed_index != None:
			selected_text = self.list_menu.confirmed_item_text
			if selected_text == 'Start SP Game':
				self.queue_state_transition(Field(self.game, 0, game_type='SP'))
			elif selected_text == 'Start MP Game':
				if self.game.connection_role == 'host':
					self.queue_state_transition(Field(self.game, 0, game_type='MP'))
				elif self.game.connection_role == 'client':
					self.queue_state_transition(Field(self.game, 1, game_type='MP'))
			elif selected_text == 'Host':
				self.queue_state_transition(HostMenu(self.game))
			elif selected_text == 'Connect':
				self.queue_state_transition(ConnectMenu(self.game))
			elif selected_text == 'Exit':
				sys.exit()

		self.list_menu.clear_confirmed()

	def draw(self):
		self.ui_container.draw(screen=self.game.screen)

	# def activate_menu(self):
	# 	selected = self.list_menu.get_selected_item()
	# 	if selected == 'Start SP Game':
	# 		self.queue_state_transition(Field(self.game, 0, 'SP'))
	# 	elif selected == 'Start MP Game':
	# 		if self.game.connected == True:
	# 			if self.game.connection_role == 'host':
	# 				self.queue_state_transition(Field(self.game, 0, 'MP'))
	# 			elif self.game.connection_role == 'client':
	# 				self.queue_state_transition(Field(self.game, 1, 'MP'))
	# 	elif selected == 'Host':
	# 		self.transition_state(HostMenu(self.game))
	# 	elif selected == 'Connect':
	# 		self.transition_state(ConnectMenu(self.game))
	# 	elif selected == 'Exit':
	# 		sys.exit()

	def keyboard_select_menu_item(self):
		return self.activate_menu()

class HostMenu(GameState):
	def __init__(self, game):
		GameState.__init__(self, game)

		self.input_map = {
			Input(key=pg.K_ESCAPE): lambda _: self.return_to_menu(),
			Input(key=pg.K_RETURN): lambda _: self._start_host()
		}

		self.port_textentry = UI.TextEntry(	pos=(c.screen_size[0]//2-100,c.screen_size[1]//2+100),
											type='port',
											font=fonts.main_menu_font_med,
											label='Port',
											default_text='4141')

		self.host_button = UI.Button(	pos=(c.screen_size[0]//2-100,c.screen_size[1]//2+200),
										font=fonts.main_menu_font_med,
										text='Host')

		self.disconnect_button = UI.Button(	pos=(c.screen_size[0]//2-100,c.screen_size[1]//2+250),
											font=fonts.main_menu_font_med,
											text='Disconnect')

		self.ui_container.add_element(self.port_textentry)
		self.ui_container.add_element(self.host_button)
		self.ui_container.add_element(self.disconnect_button)

	def _start_host(self):
		try:
			port = int(self.port_textentry.text)
		except ValueError:
			print("Invalid port")

		self.game.start_host(port)

	def _close_connection(self):
		self.game.close_connection()

	def update(self, dt, mouse_pos):
		if self.host_button.button_was_pressed == True:
			self._start_host()
			self.host_button.clear_pressed()
		elif self.disconnect_button.button_was_pressed == True:
			self._close_connection()
			self.disconnect_button.clear_pressed()

	def draw(self):
		pass

	def return_to_menu(self):
		self.queue_state_transition(MainMenu(self.game))

class ConnectMenu(GameState):
	def __init__(self, game):
		GameState.__init__(self, game)

		self.input_map = {
			Input(key=pg.K_ESCAPE): lambda _: self.return_to_menu(),
			Input(key=pg.K_RETURN): lambda _: self._submit()
		}

		self.ip_textentry = UI.TextEntry(	pos=(c.screen_size[0]//2-100,c.screen_size[1]//2),
											type='ip',
											font=fonts.main_menu_font_med,
											label='IP Address',
											default_text='localhost')

		self.port_textentry = UI.TextEntry(	pos=(c.screen_size[0]//2-100,c.screen_size[1]//2+100),
											type='port',
											font=fonts.main_menu_font_med,
											label='Port',
											default_text='4141')

		self.connect_button = UI.Button(	pos=(c.screen_size[0]//2-100,c.screen_size[1]//2+200),
											font=fonts.main_menu_font_med,
											text='Connect')

		self.ui_container.add_element(self.ip_textentry)
		self.ui_container.add_element(self.port_textentry)
		self.ui_container.add_element(self.connect_button)

		self.sel = None

	def _submit(self):
		self._attempt_to_connect(self.ip_textentry.text, int(self.port_textentry.text))

	def _attempt_to_connect(self, host, port):
		self.game._attempt_to_connect(host, port)

	def return_to_menu(self):
		self.queue_state_transition(MainMenu(self.game))

	def update(self, dt, mouse_pos):
		if self.connect_button.button_was_pressed == True:
			self._submit()
			self.connect_button.clear_pressed()

class Field(GameState):
	def __init__(self, game, player_number, game_type):
		GameState.__init__(self, game)

		self.player_number = player_number
		self.game_type = game_type
		self.board = Board(self, c.grid_count)
		self.hands = [Hand(self), Hand(self)]

		for hand in self.hands:
			hand.add_random_cards(count=2)

		self.player_turn = 0 # Player 'id' of the player whose turn it is.

		self.player_healths = [20,20]
		self.hand_area_height = 80


		self.phase = Phase(['Build','Act',"_ActAnimate"])
		self.turn_display = TurnDisplay(self.phase, fonts.ui_font)

		self.hand_origin = self.board.grid.get_grid_pos(align=('left','down'),offset=(0,50))
		self.hand_center = (c.screen_size[0]//2, c.screen_size[1]-100)
		self.hand_spacing = (110,0)
		self.drag_card = None
		self.card_grab_point = None
		self.act_animating = False
		self.act_animation_frame_count = 0


		self.player_count = 2

		# self.turn_button = UI.Button(	pos=self.board.grid.get_grid_pos(align=('left','down'),offset=(-100,-50)),
		# 							align=('right','up'),
		# 							font=fonts.ui_font,
		# 							text="End Turn",
		# 							parent_container=self)

		if player_number == 0:
			self.player_health_labels = [
			UI.Label(	pos=self.board.grid.get_grid_pos(align=('right','down'), offset=(10,-30)),
					font=fonts.ui_font,
					text='Player 0: %s'%self.player_healths[0]),
			UI.Label(	pos=self.board.grid.get_grid_pos(align=('right','up'), offset=(10,0)),
					font=fonts.ui_font,
					text='Player 1: %s'%self.player_healths[1])
			]
		elif player_number == 1:
			self.player_health_labels = [
			UI.Label(	pos=self.board.grid.get_grid_pos(align=('right','up'), offset=(10,0)),
					font=fonts.ui_font,
					text='Player 0: %s'%self.player_healths[0]),
			UI.Label(	pos=self.board.grid.get_grid_pos(align=('right','down'), offset=(10,-50)),
					font=fonts.ui_font,
					text='Player 1: %s'%self.player_healths[1])
			]

		#self.ui_container.add_element(self.turn_button)
		for label in self.player_health_labels:
			self.ui_container.add_element(label)


		self.input_map = {
			# Input(key='any'): lambda key, mod, unicode_key: self.any_key_pressed(key, mod, unicode_key),
			Input(mouse_button=1): lambda mouse_pos: self.left_mouse_pressed(mouse_pos),
			Input(mouse_button=1, type='release'): lambda mouse_pos: self.left_mouse_released(mouse_pos),
			Input(mouse_button=3): lambda mouse_pos: self.right_mouse_pressed(mouse_pos),
			Input(key=pg.K_SPACE): lambda mouse_pos: self.space_pressed(),
			Input(key=pg.K_2): lambda mouse_pos: self.hands[self.player_turn].add_card("Goblin"),
			Input(key=pg.K_DELETE): lambda mouse_pos: self.hands[self.player_turn].clear_hand(),
			Input(key=pg.K_ESCAPE): lambda mouse_pos: self.go_to_main_menu()
		}

	def change_health(self, amount, player, sync):
		if player == 0 or player == 1:
			self.player_healths[player] += amount
			self.player_health_labels[player].text = 'Player %d: %d'%(player, self.player_healths[player])

			if sync == True:
				send_string = 'health changed;' + str(amount) + ';' + str(player) + ';[END]'
				self.game.queue_network_data(send_string.encode('utf-8'))
		else:
			print("Tried to change health of invalid player.")

	@property
	def active_hand(self):
 		return self.hands[self.player_number]

	@property
	def hand_rect(self):
		card_coords = self.generate_hand_card_positions()
		if len(card_coords) == 0:
			hand_left = self.hand_center[0]
		else:
			hand_left = card_coords[0][0]
		return pg.Rect((hand_left, self.hand_center[1]), (self.active_hand.card_count*self.hand_spacing[0], c.hand_card_size[1]))

	def space_pressed(self):
		if self.game_type == 'SP' or self.is_current_player_active():
			if self.phase.name != '_ActAnimate':
				self._advance_turn(sync=True)

	@property
	def action_panel_rect(self):
		pos = self.board.grid.get_grid_pos(align=('right','center'), offset=(50, -c.action_panel_size[1]//2))
		return pg.Rect(pos, c.action_panel_size)

	def left_mouse_pressed(self, mouse_pos):
		if self.hand_rect.collidepoint(mouse_pos): # mouse is hovering hand
			hand_left = self.get_left_side_of_hand()
			relative_x = int((mouse_pos[0] - hand_left) % self.hand_spacing[0])
			relative_y = int(mouse_pos[1] - self.hand_center[1])
			clicked_card_index = int((mouse_pos[0] - hand_left) // self.hand_spacing[0])

			if relative_x >= 0 and relative_x < c.hand_card_size[1]:
				if clicked_card_index >= 0 and clicked_card_index < self.active_hand.card_count:
					self.drag_card = self.active_hand.pop_card(clicked_card_index)
					# d.debugger.print(values={'drag_card': self.drag_card})
					self.card_grab_point = (relative_x,relative_y)

		if self.board.grid.rect.collidepoint(mouse_pos): # mouse is hovering board
			self.board.selected_cell = self.board.grid.get_cell_at_pos(pos=mouse_pos)
		else:
			self.board.selected_cell = None

	def is_current_player_active(self):
		if self.player_turn == self.player_number:
			return True
		else:
			return False

	def left_mouse_released(self, mouse_pos):
		if self.drag_card is not None:
			placed_in_board = False # True if card is placed onto the board during this mouse release

			if self.is_current_player_active() and self.phase.name == 'Build':
				cell = self.board.grid.get_cell_at_pos(pos=mouse_pos)
				if cell != None:
					# d.debugger.print(values={'drag_card': self.drag_card})
					self.drag_card.queue(board=self.board, cell=cell, owner=self.player_number)
					placed_in_board = True

			if placed_in_board == False:
				self.active_hand.add_card(name=self.drag_card.name)

			self.drag_card = None
			self.card_grab_point = None # Probably not necessary

		# TODO: Move this to the proper place (the event queue?)
		# if self.turn_button.left_mouse_released(mouse_pos=mouse_pos): # if button was pressed
		# 	if self.game_type == 'SP':
		# 		self._end_turn(sync=True)
		# 	elif self.game_type == 'MP':
		# 		if self.is_current_player_active():
		# 			self._end_turn(sync=True)

	def right_mouse_pressed(self, mouse_pos):
		# If a cell is selected
		if self.board.grid.rect.collidepoint(mouse_pos):
			if self.board.selected_cell != None:
				selected_building = self.board.building_cards[self.board.selected_cell]
				if selected_building != None and selected_building.owner == self.player_number and self.is_current_player_active():
					clicked_cell = self.board.grid.get_cell_at_pos(pos=mouse_pos)
					selected_building.target_cell = clicked_cell
					selected_building.act()

	def set_active_player(self, player_number):
		self.player_turn = player_number
		self._end_turn(sync=True)

	def swap_active_player(self):
		if self.player_turn == 0:
			self.player_turn = 1
		elif self.player_turn == 1:
			self.player_turn = 0

	def _end_turn(self, sync):
		end_of_turn = self._advance_turn(sync=sync)
		while end_of_turn == False:
			end_of_turn = self._advance_turn(sync=sync)

	def start_act_animations(self):
		self.act_animating = True
		self.act_animation_frame_count = 0

	def _advance_turn(self, sync):
		if self.phase.name == 'Act':
			for lane in range(self.board.lane_count):
				for rank in range(self.board.rank_count):
					cell = (lane,rank)
					if self.player_turn == 0:
						absolute_cell = cell
					else:
						absolute_cell = (cell[0], self.board.rank_count-1-cell[1]) # Corrected for player orientation (actual grid coords)

					building_card = self.board.building_cards[absolute_cell]
					unit_card = self.board.get_unit_in_cell(cell=absolute_cell)

					if building_card != None and building_card.owner == self.player_turn:
						building_card.act()

					if unit_card != None and unit_card.owner == self.player_turn:
						unit_card.act()

			# Transfer queued cards to battlefield in closest rank
			for lane_number, queued_card in enumerate(self.board.queued_cards[self.player_turn]):
				if queued_card != None:
					if self.player_turn == 0:
						cell = (lane_number, self.board.size[1]-1)
					elif self.player_turn == 1:
						cell = (lane_number, 0)

					queued_card.place_queued()
					#self.board.place_card(cell=cell, card=queued_card, owner=self.player_turn, sync=sync)
					self.board.queued_cards[self.player_number][lane_number] = None

			self.board.refresh_fow()
			self.act_animating = True
			self.act_animation_frame_count = 0

		if sync == True:
			send_string = 'phase advanced' + ';[END]'
			self.game.queue_network_data(send_string.encode('utf-8'))

		self.phase.advance_phase()

		if self.phase.turn_ended == True:
			for card in self.board:
				if card is not None:
					card.end_turn()

			self.phase.end_turn()
			self.swap_active_player()
			self.hands[self.player_turn].add_random_cards(count=1)

			if self.game_type == 'SP' and self.player_turn == 1:
				# Single player and it's the computer's turn.
				# For now, we just want to play 1 random card from their hand to a random square on the board and end their turn.
				hand = self.hands[self.player_turn]
				card = hand.cards.pop(random.randrange(0,hand.card_count))
				cell_x = random.randrange(0, self.board.size[0])
				cell_y = random.randrange(0, self.board.size[1])

				dont_queue = False
				if isinstance(card, BuildingCard):
					if self.board.building_cards[(cell_x,cell_y)] is not None:
						dont_queue = True

				if dont_queue is False:
					card.queue(board=self.board, cell=(cell_x,cell_y), owner=1)
			return True
		else:
			return False

	def process_network_data(self, data):
		raw_data_string = data.decode('utf-8')
		event_strings = raw_data_string.split(';[END]')
		for event_string in event_strings:
			print('event_string=', event_string)
			args = event_string.split(';')
			try:
				if args[0] == 'card placed':
					# This doesn't have permanence in card values (like creature health), since it only sends the card name
					# However, maybe this is ok, if this is only used when the card is played from hand or moved on from lane queue
					self.board.place_card(cell=(int(args[2]),int(args[3])), card=self.game.card_pool.get_card_by_name(args[1]), owner=args[4], sync=False)
				elif args[0] == 'unit deleted':
					self.board.delete_unit_from_board(cell=(int(args[1]), int(args[2])), sync=False)
				elif args[0] == 'building deleted':
					self.board.delete_building_from_board(cell=(int(args[1]), int(args[2])), sync=False)
				elif args[0] =='unit moved':
					self.board.move_unit(start_cell=(int(args[1]), int(args[2])), target_cell=(int(args[3]),int(args[4])), sync=False)
				elif args[0] =='building moved':
					self.board.move_building(start_cell=(int(args[1]), int(args[2])), target_cell=(int(args[3]),int(args[4])), sync=False)
				elif args[0] == 'card removed':
					self.board.return_card_to_hand((int(args[2]), int(args[3])))
				elif args[0] == 'cards fought':
					self.board.fight_cards(attacker_cell=(int(args[1]), int(args[2])), defender_cell=(int(args[3]), int(args[4])), sync=False)
				elif args[0] == 'turn ended':
					self._end_turn(sync=False)
				elif args[0] == 'phase advanced':
					self._advance_turn(sync=False)
				elif args[0] == 'health changed':
					self.change_health(amount=int(args[1]), player=int(args[2]), sync=False)
				elif args[0] == 'message sent':
					self.game.chat_window.add_message(args[1], args[2])
			except:
				print('Invalid arguments for network event string')

	def go_to_main_menu(self):
		send_string = 'quit field' + ';[END]'
		self.game.queue_network_data(send_string.encode('utf-8'))
		self.queue_state_transition(MainMenu(self.game))

	def generate_hand_card_positions(self):
		total_width = self.active_hand.card_count * self.hand_spacing[0]
		return [(self.hand_center[0]+i*self.hand_spacing[0]-total_width//2,self.hand_center[1]) for i in range(self.active_hand.card_count)]

	def get_left_side_of_hand(self):
		card_coords = self.generate_hand_card_positions()
		if len(card_coords) == 0:
			return self.hand_center[0]
		else:
			return card_coords[0][0]

	def _generate_hovered_card_index(self, mouse_pos):
		if self.hand_rect.collidepoint(mouse_pos):
			for i, card_pos in enumerate(self.generate_hand_card_positions()):
				card_left = card_pos[0]
				card_right = card_pos[0] + c.hand_card_size[0]

				if mouse_pos[0] >= card_left and mouse_pos[0] < card_right:
					self.hovered_card_index = i
					return

		self.hovered_card_index = None

	def get_action_icon_pos(self, action_index):
		row = i // pan

	def _draw_action_panel(self):
		action_panel_surface = pg.Surface(c.action_panel_size)
		action_panel_surface.fill(c.black)

		icon_width, icon_height = c.action_icon_size
		panel_width, panel_height = c.action_panel_size
		panel_count_x = panel_width // icon_width
		panel_count_y = panel_height // icon_height

		if self.board.selected_cell != None:
			selected_card = self.board.building_cards[self.board.selected_cell]
			if selected_card != None:
				for i, action in enumerate(selected_card.active_actions):
					row = i // panel_count_x
					col = i % panel_count_y

					cell_pos = (col*icon_width, row*icon_height)
					pg.draw.rect(action_panel_surface, c.dark_grey, (cell_pos, c.action_icon_size))
					pg.draw.rect(action_panel_surface, c.light_grey, (cell_pos, c.action_icon_size), 1)
					draw.draw_surface_aligned(	target=action_panel_surface,
											source=fonts.action_font.render(action, True, c.white),
											pos=(cell_pos[0]+icon_width//2, cell_pos[1]+icon_height//2),
											align=('center','center'))

		pg.draw.rect(action_panel_surface, c.white, ((0,0),c.action_panel_size), 1)
		draw.screen.blit(action_panel_surface, self.action_panel_rect.topleft)

	def update(self, dt, mouse_pos):
		self._generate_hovered_card_index(mouse_pos)
		if self.act_animating is True:
			self.act_animation_frame_count += 1

			if self.act_animation_frame_count > 60:
				self.act_animating = False
				self._advance_turn(sync=False)

	def draw(self):
		if self.player_number == 0:
			current_player_color = c.red
			other_player_color = c.blue
		elif self.player_number == 1:
			current_player_color = c.blue
			other_player_color = c.red

		if self.player_turn == 0:
			active_player_color = c.red
		elif self.player_turn == 1:
			active_player_color = c.blue

		# Draw board
		if self.act_animating:
			t = (self.act_animation_frame_count / 60)
			self.board.draw(screen=self.game.screen, player_perspective=self.player_number, animation_interp_factor=t)
		else:
			self.board.draw(screen=self.game.screen, player_perspective=self.player_number)

		# Draw queued cards
		for lane_number, queued_card in enumerate(self.board.queued_cards[self.player_number]):
			if queued_card != None:
				pos = self.board.grid.get_cell_pos(cell=(lane_number, self.board.size[1]-1), align=('left','down'))
				pos[0] += c.board_card_size[0]
				queued_card.draw(pos=pos, location='queue')

		self._draw_action_panel()
		if self.board.selected_cell != None:
			self.board.building_cards[self.board.selected_cell]

		# pg.draw.rect(self.action_panel_surface, c.white, ((0,0), c.action_panel_size), 1)

		# Calculate colors for card areas based on which player is active
		if self.is_current_player_active():
			my_card_area_color = util.darken_color(current_player_color,0.65)
			other_card_area_color = c.dark_grey
		else:
			my_card_area_color = c.dark_grey
			other_card_area_color = util.darken_color(other_player_color,0.65)

		# Draw my card area
		pg.draw.rect(	draw.screen, my_card_area_color,
						((0,c.screen_size[1]-self.hand_area_height),
						(c.screen_size[0], self.hand_area_height))
					)
		pg.draw.line(	draw.screen, util.lighten_color(my_card_area_color,0.5),
						(0,c.screen_size[1]-self.hand_area_height),
						(c.screen_size[0],c.screen_size[1]-self.hand_area_height)
					)
		# Draw enemy card area
		pg.draw.rect(	draw.screen, other_card_area_color,
						((0,0), (c.screen_size[0], self.hand_area_height)))
		pg.draw.line(	draw.screen, util.lighten_color(other_card_area_color,0.5),
						(0, self.hand_area_height),
						(c.screen_size[0], self.hand_area_height))


		# Draw cards in hand
		for i, card_pos in enumerate(self.generate_hand_card_positions()):
			if i == self.hovered_card_index:
				hover = True
			else:
				hover = False

			self.active_hand[i].draw(pos=card_pos, location='hand', hover=hover)

		# Draw active player text and circle
		active_player_text = "Player %d"%self.player_turn
		text_h_padding = 10
		text_size = fonts.ui_font.size(active_player_text)
		padded_size = (text_size[0]+2*text_h_padding, text_size[1])
		active_player_text_surface = pg.Surface(padded_size)
		pg.draw.rect(active_player_text_surface, c.white, ((0,0),(padded_size)))
		active_player_text_surface.blit(fonts.ui_font.render(active_player_text, True, active_player_color), (text_h_padding,0))
		draw_pos = (20, c.screen_size[1]//2)
		offset = draw.draw_surface_aligned(	target=draw.screen,
										source=active_player_text_surface,
										pos=draw_pos,
										align=('left','down'))
		pg.draw.circle(draw.screen, active_player_color,
						(	int(draw_pos[0] + offset[0] + active_player_text_surface.get_width() + 20),
							int(draw_pos[1] + offset[1] + active_player_text_surface.get_height()//2)),
						15)
		pg.draw.circle(draw.screen, c.white,
						(	int(draw_pos[0] + offset[0] + active_player_text_surface.get_width() + 20),
							int(draw_pos[1] + offset[1] + active_player_text_surface.get_height()//2)),
						15, 1)

		# Draw turn display
		self.turn_display.draw(target=draw.screen, pos=self.board.grid.get_grid_pos(align=('left','center'),offset=(-150,0)))

		# Draw card being dragged
		if self.drag_card:
			drawn_in_board = False # True if the drag card gets drawn in the board this frame rather than floating on screen

			if self.is_current_player_active() and self.phase.name == 'Build':
				if isinstance(self.drag_card, CreatureCard):
					# TODO: Shouldn't be getting the mouse position in this way. Fetch it in update()
					cell = self.board.grid.get_cell_at_pos(pos=pg.mouse.get_pos())
					if cell != None:
						lane = cell[0]
						lane_down_center = self.board.grid.get_cell_pos(cell=(lane, self.board.size[1]-1), align=('left','down'))
						lane_down_center[0] += c.board_card_size[0]
						self.drag_card.draw(pos=lane_down_center, location="board_hover")
						drawn_in_board = True
				elif isinstance(self.drag_card, BuildingCard):
					cell = self.board.grid.get_cell_at_pos(pos=pg.mouse.get_pos())
					if cell != None:
						if self.player_number == 0:
							pos = cell
						elif self.player_number == 1:
							pos = (cell[0],self.board.size[1]-1-cell[1])

						if self.board.building_cards[pos] == None:
							cell_top_center = self.board.grid.get_cell_pos(cell, align=('left','top'))
							self.drag_card.draw(cell_top_center, "board_hover")
							drawn_in_board = True

			if drawn_in_board == False:
				mouse_coords = pg.mouse.get_pos()
				pos = (mouse_coords[0] - self.card_grab_point[0], mouse_coords[1] - self.card_grab_point[1])
				self.drag_card.draw(pos, "hand")
