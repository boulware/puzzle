import sys, os, copy, collections
from enum import Enum, auto, IntEnum
import pygame as pg
from pygame.math import Vector2 as Vec
import numpy as np
from functools import partial

# General constants
black = (0,0,0)
grey = (127,127,127)
light_grey = (200,200,200)
dark_grey = (30,30,30)
white = (255,255,255)
red = (255,0,0)
dark_red = (70,0,0)
green = (0,255,0)
blue = (0,0,255)
dark_blue = (0,0,70)

# Game parameters
grid_count = (5,6)
node_size = (90,90)
grid_origin = (100,10)

class Node: 
	def __init__(self, position, strength):
		self.position = position
		self.strength = strength

	def draw(self):
		color_cell(self.position, light_grey)
		strength_text = node_font.render(str(self.strength), True, red)


		#pg.draw.circle(screen, red, pos_to_coords_center(self.position), 1)
		cell_center = pos_to_coords_center(self.position)
		screen.blit(strength_text, (cell_center[0] - 0.5*strength_text.get_width(), cell_center[1] - 0.5*strength_text.get_height()))

# Returns a tuple of four sets of coordinates, which are guaranteed to either be a neighbor of (position) or (position) itself
def get_neighbors(position):
	neighbors = []
	candidates = []

	candidates.append((max(0, position[0]-1), position[1]))
	candidates.append((min(grid_count[0]-1, position[0]+1), position[1]))
	candidates.append((position[0], max(0, position[1]-1)))
	candidates.append((position[0], min(grid_count[1]-1, position[1]+1)))

	for candidate in candidates:
		if candidate != position:
			neighbors.append(candidate)

	return neighbors

class GameGrid:
	def __init__(self, size):
		self.width = size[0]
		self.height = size[1]
		self.cells = np.zeros((self.width, self.height), np.int8)

	def set_cell(self, pos, strength):
		np.put(self.cells, pos, strength)

	def increment_cell(self, pos, amount):
		self.cells[pos] += amount

	def draw_strengths(self):
		it = np.nditer(self.cells, flags=['multi_index'])
		while not it.finished:
			color = grey
			if it[0] > 0: color = green
			if it[0] < 0: color = red

			strength_text = node_font.render(str(it[0]), True, color)
			cell_center = pos_to_coords_center(it.multi_index)
			screen.blit(strength_text, (cell_center[0] - 0.5*strength_text.get_width(), cell_center[1] - 0.5*strength_text.get_height()))			

			it.iternext()

class Grid:
	def __init__(self, dimensions, origin, cell_size):
		self.dimensions = np.array(dimensions)
		self.rect = pg.Rect(origin, [(dimensions[x_n] * cell_size[x_n]) for x_n in range(2)])
		self.cell_size = cell_size
		self.update_drawable()

	def update_drawable(self):
		self.drawable = True
		for dim in self.dimensions:
			if dim <= 0:
				self.drawable = False

	def resize(self, amounts):
		if self.dimensions.size != len(amounts):
			print("Invalid size given to resize()")
			return False

		self.dimensions = np.add(self.dimensions, amounts)

		self.update_drawable()

		return True

	def get_cells_by_distance(self, start_cell, distance): # Using Chebyshev distance (King's distance)
		distance = max(0, distance) # Clamp distance to a non-negative value

		start_x = start_cell[0]
		min_x = max(0, start_x - distance)
		max_x = min(self.dimensions[0]-1, start_x + distance)

		start_y = start_cell[1]
		min_y = max(0, start_y - distance)
		max_y = min(self.dimensions[1]-1, start_y + distance)

		return [(x,y) for x in range(min_x,max_x+1) for y in range(min_y,max_y+1)]

	# Return the cell position. Align lets you choose among the corners, centers of edges, or center. Default params top left corner
	def get_cell_pos(self, grid_coords, align=('left','up')):
		pos = [self.rect[i] + grid_coords[i]*self.cell_size[i] for i in range(2)]
		
		if align[0] == 'center':
			pos[0] += self.cell_size[0]//2
		elif align[0] == 'right':
			pos[0] += self.cell_size[0]

		if align[1] == 'center':
			pos[1] += self.cell_size[1]//2
		elif align[1] == 'down':
			pos[1] += self.cell_size[1]

		return pos

	# Return the grid position in screen space. Align lets you choose among the corners, centers of edges, or center. Default params give top left corner
	def get_grid_pos(self, align=('left','up'), offset=(0,0)):
		pos = list(self.rect.topleft)

		if align[0] == 'center':
			pos[0] += self.rect.width//2
		elif align[0] == 'right':
			pos[0] += self.rect.width

		if align[1] == 'center':
			pos[1] += self.rect.height//2
		elif align[1] == 'down':
			pos[1] += self.rect.height

		pos[0] += offset[0]
		pos[1] += offset[1]

		return pos

	def get_cell_at_mouse(self):
		hit = False
		mouse_x, mouse_y = pg.mouse.get_pos()

		grid_x = (mouse_x - self.rect.x) // self.cell_size[0]
		grid_y = (mouse_y - self.rect.y) // self.cell_size[1]

		if grid_x >=0 and grid_x < self.dimensions[0] and grid_y >= 0 and grid_y < self.dimensions[1]:
			hit = True

		return {'hit': hit, 'pos': (grid_x, grid_y)}

	def draw(self, color=white):
		if self.drawable:
			for x in range(self.dimensions[0] + 1):
				x_start = self.rect.x + x*self.cell_size[0]
				pg.draw.line(screen, color, (x_start, self.rect.y), (x_start, self.rect.y + self.cell_size[1]*self.dimensions[1]))
			for y in range(self.dimensions[1] + 1):
				y_start = self.rect.y + y*self.cell_size[1]
				pg.draw.line(screen, color, (self.rect.x, y_start), (self.rect.x + self.cell_size[0]*self.dimensions[0], y_start))

	def color_cell(self, position, color):
		cell_rect = self.get_cell_rect(position)
		cell_rect.inflate_ip((-2,-2))

		pg.draw.rect(screen, color, cell_rect)

	def get_cell_rect(self, position):
		return pg.Rect(self.get_cell_pos(position), np.add(self.cell_size, (1,1)))

	def check_cell_valid(self, cell):
		if cell[0] < 0 or cell[0] > self.dimensions[0]-1 or cell[1] < 0 or cell[1] > self.dimensions[1]-1:
			return False
		else:
			return True

	def draw_surface_in_cell(self, source, grid_coords, align=('left','up'), stretch=False, offset=(0,0)):
		surface = source

		if stretch == True:
			width_ratio = source.get_width() / self.cell_size[0]
			height_ratio = source.get_height() / self.cell_size[1]

			if width_ratio != 1 or height_ratio != 1:
				width_ratio_distance = (1 - width_ratio) # "distance" from 1
				height_ratio_distance = (1 - height_ratio)

				if abs(width_ratio_distance) > abs(height_ratio_distance):
					scale_ratio = width_ratio # Scale width to cell width, retaining aspect ratio
				else:
					scale_ratio = height_ratio # Scale height

				new_width = int(source.get_width()/scale_ratio)
				new_height = int(source.get_height()/scale_ratio)
				surface = pg.transform.scale(source, (new_width,new_height)) # new scaled surface

		cell_pos = self.get_cell_pos(grid_coords, align)
		draw_surface_aligned(target=screen, source=surface, pos=cell_pos, align=align, offset=offset)

def draw_surface_aligned(target, source, pos, align=('left','left'), offset=(0,0)):
	new_pos = list(np.add(pos, offset))

	if align[0] == 'center':
		new_pos[0] -= source.get_width()//2
	elif align[0] == 'right':
		new_pos[0] -= source.get_width()

	if align[1] == 'center':
		new_pos[1] -= source.get_height()//2
	elif align[1] == 'down':
		new_pos[1] -= source.get_height()

	target.blit(source, new_pos)

hand_card_size = (100,160)
board_card_size = (56,90)

CreatureStats = collections.namedtuple('CreatureStats', 'power health')

class Card:
	def __init__(self, name, cost, begin_phase_fns=[], attack_phase_fns=[], passive_fns=[]):
		self.name = name
		self.cost = cost
		self.pos = None
		self._owner = None

		self.begin_phase_fns = begin_phase_fns
		self.attack_phase_fns = attack_phase_fns
		self.passive_fns = passive_fns
		self.buffs = []

		self.active = False

		self.generate_surfaces()

	def __generate_hand_surface(self):
		bg_color = dark_grey
		if self.owner == 0:
			bg_color = dark_red
		elif self.owner == 1:
			bg_color = dark_blue

		self.hand_surface = pg.Surface(hand_card_size)
		pg.draw.rect(self.hand_surface, bg_color, ((0,0), hand_card_size))
		pg.draw.rect(self.hand_surface, light_grey, ((0,0), hand_card_size), 1)
		title_surface = card_text_sm.render(self.name, True, white)
		self.hand_surface.blit(title_surface, (5,0))
		cost_surface = card_text_lg.render(str(self.cost), True, grey)
		draw_surface_aligned(target=self.hand_surface, source=cost_surface, pos=self.hand_surface.get_rect().center, align=('center','center'))

	def __generate_board_surface(self):
		self.board_surface = pg.transform.smoothscale(self.hand_surface, board_card_size)

	@property
	def enemy(self):
		print("pos=", self.pos, "name=", self.name, "owner=", self.owner)
		if self.owner == 0:
			return 1
		elif self.owner == 1:
			return 0
		else:
			return None

	def generate_surfaces(self):
		self.__generate_hand_surface()
		self.__generate_board_surface()

	@property
	def owner(self):
		return self._owner

	@owner.setter
	def owner(self, owner):
		self._owner = owner
		self.generate_surfaces()

	def clone(self):
		return Card(name = self.name,
					cost = self.cost,
					begin_phase_fns = copy.deepcopy(self.begin_phase_fns),
					attack_phase_fns = copy.deepcopy(self.attack_phase_fns),
					passive_fns = copy.deepcopy(self.passive_fns))

	def apply_buff(self):
		pass

	def clear_buffs(self):
		self.buffs=[]
		self.generate_surfaces()

	def do_passive(self):
		if self.active:
			for fn in self.passive_fns:
				fn(self)

	def do_begin_phase(self):
		if self.active:
			for fn in self.begin_phase_fns:
				fn(self)

	def do_attack_phase(self):
		if self.active:
			for fn in self.attack_phase_fns:
				fn(self)

	def draw(self, pos, location):
		if location == "hand":
			screen.blit(self.hand_surface, pos)
		if location == "board" or location == "board_hover":
			screen.blit(self.board_surface, pos)

		if self.active == False and location == "board":
			pg.draw.line(screen, red, pos, (pos[0]+board_card_size[0], pos[1]+board_card_size[1]))

class CreatureCard(Card):
	def __init__(self, name, cost, base_power, base_health, begin_phase_fns=[], attack_phase_fns=[], passive_fns=[]):
		self.base_power = base_power
		self.base_health = base_health

		Card.__init__(self=self, name=name, cost=cost, begin_phase_fns=begin_phase_fns, attack_phase_fns=attack_phase_fns)

	@property
	def power(self):
		return self.get_stats().power

	@property
	def health(self):
		return self.get_stats().health

	def __generate_hand_surface(self):
		self._Card__generate_hand_surface()
		stats = self.get_stats()
		stats_surface = card_text_sm.render(str(stats[0]) + '/' + str(stats[1]), True, green)
		bottomright = self.hand_surface.get_rect().bottomright
		draw_surface_aligned(	target=self.hand_surface, 
								source=stats_surface,
								pos=bottomright,
								align=('right','down'),
								offset=(-2,-2))

	def __generate_board_surface(self):
		self._Card__generate_board_surface()

	def generate_surfaces(self):
		self.__generate_hand_surface()
		self.__generate_board_surface()

	def apply_buff(self, power=0, health=0):
		buff = (power,health)
		if buff != (0,0):
			self.buffs.append(buff)

		self.generate_surfaces()

	def do_attack_phase(self):
		Card.do_attack_phase(self)
		if board.check_if_card_is_front(self.pos) == True:
			print("changing health of player[", self.enemy, "]")
			game.change_health(-self.power, self.enemy)

	def get_stats(self):
		power = self.base_power
		health = self.base_health

		for buff in self.buffs:
			power += buff[0]
			health += buff[1]

		return CreatureStats(power=power, health=health)

	def clone(self):
		return CreatureCard(name = self.name,
							cost = self.cost,
							base_power = self.base_power,
							base_health = self.base_health,
							begin_phase_fns = copy.deepcopy(self.begin_phase_fns),
							attack_phase_fns = copy.deepcopy(self.attack_phase_fns),
							passive_fns = copy.deepcopy(self.passive_fns))

class CardPool:
	def __init__(self):
		self.names = ['null']
		self.surfaces = [None]
		self.cards = [None]

		self.invalid_surface = node_font.render('?', True, red)
		self.invalid_card = Card([])

	def add_card(self, name, surface, card):
		result = self.get_id_by_name(name)
		if result['exists'] == True:
			print("Tried to add item to item pool with duplicate name. Item not added.")
			return {'success': False, 'ID': None}

		self.names.append(name)
		self.surfaces.append(surface)
		self.cards.append(card)

		return {'success': True, 'ID': len(self.names)-1}

	def card_exists(self, ID):
		if ID < len(self.names) and ID > 0: # Don't include 'null' item as a valid item (ID > 0)
			return True
		else:
			return False

	def get_card_by_id(self, ID):
		if self.card_exists(ID):
			return self.cards[ID]
		else:
			return self.invalid_card

	def get_surface_by_id(self, ID):
		if self.card_exists(ID):
			return self.surfaces[ID]
		else:
			return self.invalid_surface

	def get_id_by_name(self, name):
		for i, item_name in enumerate(self.names):
			if item_name == name and item_name != 'null': # Don't include 'null' item as an item
				return {'exists': True, 'ID': i}

		return {'exists': False, 'ID': None}

	def get_name_by_id(self, ID):
		if self.card_exists(ID):
			return {'exists:': True, 'name': self.names[ID]}
		else:
			print("Tried to reference card with non-existent name or null card.")
			return {'exists': False, 'name': None}

class Hand:
	def __init__(self):
		self.cards = []

		self.origin = Vec(10,620)
		self.card_spacing = 110

		self.drag_card = None
		self.card_grab_point = None

	def add_card(self, card, count=1):
		if not isinstance(card, Card):
			print("Tried to add card to hand that wasn't Card or a subclass.")
			return
		if count <= 0:
			return

		for i in range(count):
			self.cards.append(card.clone())

	def clear_hand(self):
		self.cards = []

	def mouse_press(self, pos):
		mouse_x, mouse_y = pos[0], pos[1]
		for i in range(len(self.cards)):
			left_x = self.origin[0] + (self.card_spacing * i)
			right_x = left_x + hand_card_size[0]
			top_y = self.origin[1]
			bottom_y = self.origin[1] + hand_card_size[1]

			if mouse_x >= left_x and mouse_x <= right_x and mouse_y > top_y and mouse_y < bottom_y:
				# Mouse click happened on a card (self.cards[i])
				if not self.drag_card:
					self.drag_card = self.cards.pop(i)
					# self.drag_card_index = i
					self.card_grab_point = Vec(mouse_x - left_x, mouse_y - top_y)

	def mouse_release(self, pos):
		if self.drag_card:
			placed_in_board = False # True if card is placed onto the board during this mouse release

			result = board.grid.get_cell_at_mouse()
			if result['hit'] == True: # If the mouse is hovering over somewhere on the board grid while dragging a card
				pos = result['pos']
				if board.cards[pos] == None:
					placed_in_board = board.place_card(result['pos'], self.drag_card)
			
			if placed_in_board == False:
				self.cards.append(self.drag_card)
			
			self.drag_card = None
			self.card_grab_point = None # Probably not necessary 

	def draw(self):
		# Draw each card in the hand with horizontal spacing between them
		for i, card in enumerate(self.cards):
			card.draw(self.origin + Vec(i*self.card_spacing,0), "hand")

		# Draw the drag card in a grid cell if the mouse is hovering over it
		if self.drag_card:
			drawn_in_board = False # True if the drag card gets drawn in the board this frame rather than floating on screen

			result = board.grid.get_cell_at_mouse()
			if result['hit'] == True: # If the mouse is hovering over somewhere on the board grid while dragging a card
				pos = result['pos']
				if board.cards[pos] == None:
					cell_top_left = board.grid.get_cell_pos(result['pos'], align=('center','top'))
					cell_top_left[0] -= board_card_size[0]//2
					self.drag_card.draw(cell_top_left, "board_hover")
					drawn_in_board = True
			
			if drawn_in_board == False:
				mouse_coords = Vec(pg.mouse.get_pos())
				self.drag_card.draw(mouse_coords - self.card_grab_point, "hand")

Phases = {	"Begin":	0,
			"Attack":	1,
			"End":	 	2,
			0: "Begin",
			1: "Attack",
			2: "End"}

line_spacing = 9

class TurnDisplay:
	def __init__(self):
		self.phase = "Begin"
		self.phase_text = None
		self.__generate_phase_text()

	def set_active_phase(self, phase):
		self.phase = phase
		self.__generate_phase_text()

	def __generate_phase_text(self):
		begin_color, attack_color, end_color = white, white, white

		if self.phase == "Begin":
			begin_color = green
		elif self.phase == "Attack":
			attack_color = green
		elif self.phase == "End":
			end_color = green

		begin_text = ui_font.render("Begin", True, begin_color)
		attack_text = ui_font.render("Attack", True, attack_color)
		end_text = ui_font.render("End", True, end_color)

		if self.phase_text == None:
			sum_heights = 24*3 + line_spacing*2
			max_width = max(begin_text.get_width(), attack_text.get_width(), end_text.get_width())
			self.phase_text = pg.Surface((max_width, sum_heights + line_spacing*2))
			self.phase_text.set_colorkey(black)
		else:
			self.phase_text.fill(black)

		self.phase_text.blit(begin_text, (0,0))
		self.phase_text.blit(attack_text, (0,24+line_spacing*1))
		self.phase_text.blit(end_text, (0,48+line_spacing*2))

	def draw(self, pos, align):
		draw_surface_aligned(target=screen, source=self.phase_text, pos=pos, align=align)

class Game:
	def __init__(self):
		self.turn_display = TurnDisplay()

		self._turn_number = 0
		self.__start_turn()
		self.__refresh_turn_surface()

		self.player_healths = [20,20]
		self.__refresh_hp_surfaces()
		self.hp_text_offset = (10,0)

	def __refresh_turn_surface(self):
		self.turn_text = ui_font.render("Turn: " + str(self._turn_number) + '(' + self._phase_name + ')', True, white)

	def __refresh_hp_surfaces(self):
		self._bottom_hp_text = ui_font.render("HP: " + str(self.player_healths[0]), True, white)
		self._top_hp_text = ui_font.render("HP: " + str(self.player_healths[1]), True, grey)

	def is_valid_player(self, player):
		if player == 0 or player == 1:
			return True
		else:
			return False

	def change_health(self, amount, player):
		if self.is_valid_player(player):
			self.player_healths[player] += amount
			self.__refresh_hp_surfaces()
		else:
			print("Tried to change health of invalid player.")

	def __start_turn(self):
		self._phase_name = "Begin"
		self._phase_number = Phases[self._phase_name]
		self.turn_display.set_active_phase(self._phase_name)

	def __advance_phase(self):
		# We do a blind advance phase, and rely on something else to fix it if it goes past the end phase
		self._phase_number += 1
		self._phase_name = Phases[self._phase_number]
		self.turn_display.set_active_phase(self._phase_name)

	def advance_turn(self):
		if self._phase_name == "Begin":
			board.do_begin_phase()
			self.__advance_phase()
		elif self._phase_name == "Attack":
			board.do_attack_phase()
			self.__advance_phase()
		elif self._phase_name == "End":
			self._turn_number += 1
			self.__start_turn()

		self.__refresh_turn_surface()

	def draw(self):
		self.turn_display.draw(board.grid.get_grid_pos(align=('right','center'),offset=(50,0)), align=('left','center'))
		draw_surface_aligned(target=screen, source=self._bottom_hp_text, pos=board.grid.get_grid_pos(align=('right','down')), align=('left','down'), offset=self.hp_text_offset)
		draw_surface_aligned(target=screen, source=self._top_hp_text, pos=board.grid.get_grid_pos(align=('right','up')), offset=self.hp_text_offset)

class Board:
	def __init__(self, size):
		self.size = size
		self.cards = np.full(size, None, np.dtype(Card))
		self.grid = Grid(dimensions=size, origin=(10,10), cell_size=node_size)
		self.__reset_mana()

	def place_card(self, cell, card):
		if self.grid.check_cell_valid(cell) == True:
			card.pos = cell
			card.owner = board.get_cell_owner(cell)
			self.cards[cell] = card
			self.__refresh_passives()

			return True # Successfully fulfilled requirements for placing the card and placed it.
		else:
			print("Tried to place card in invalid cell")
			return False

	def return_card_to_hand(self, cell):
		if self.cards[cell] != None:
			self.cards[cell].owner = None
			hand.add_card(self.cards[cell])
			self.cards[cell] = None
			self.__refresh_passives()

			return True # Card returned
		else:
			return False # No card returned

	def right_mouse_press(self, pos):
		result = self.grid.get_cell_at_mouse()
		if result['hit'] == True:
			cell = result['pos']
			self.return_card_to_hand(cell)
			self.__refresh_passives()

	def get_cell_owner(self, cell):
		if self.grid.check_cell_valid(cell) == True:
			if cell[1] < grid_count[1]//2:
				return 1
			else:
				return 0
		else:
			print("Tried to get cell owner of invalid cell")
			return None


	def check_if_card_is_front(self, cell):
		if self.grid.check_cell_valid(cell) == False:
			print("check_if_card_is_front() got invalid cell")
			return

		owner = self.get_cell_owner(cell)
		col = cell[0]

		if owner == 0:
			for row in range(self.size[1]//2, cell[1]):
				if self.cards[col,row] != None:
					return False
		elif owner == 1:
			for row in range(self.size[1]//2, cell[1], -1):
				if self.cards[col,row] != None:
					return False

		return True

	def __reset_mana(self):
		self.red_mana = np.zeros(self.size, dtype=np.uint8)

	def __refresh_passives(self):
		dirty = False

		for cell, card in np.ndenumerate(self.cards):
			if card != None:
				if self.red_mana[cell] >= card.cost and card.active != True:
					dirty = True
					card.active = True
				elif self.red_mana[cell] < card.cost and card.active != False:
					dirty = True
					card.active = False

		self.__reset_mana()
		self.__clear_buffs()
		self.do_passive()

		if dirty == True:
			self.__refresh_passives() # Iterative refreshes when state has changed

	def __clear_buffs(self):
		for _, card in np.ndenumerate(self.cards):
			if card != None:
				card.clear_buffs()

	def add_mana(self, amount, type, pos, distance=1):
		if pos:
			if type == "red":
				cell_coords = self.grid.get_cells_by_distance(pos, distance)
				for cell_coord in cell_coords:
					self.red_mana[cell_coord] += amount

	def buff_creatures_in_range(self, power, health, pos, distance=1):
		if pos:
			cell_coords = self.grid.get_cells_by_distance(pos, distance)
			for cell_coord in cell_coords:
				if isinstance(self.cards[cell_coord], CreatureCard):
					self.cards[cell_coord].apply_buff(power=1,health=1)

	def do_passive(self):
		for _, card in np.ndenumerate(self.cards):
			if card != None:
				card.do_passive()

	def do_begin_phase(self):
		for _, card in np.ndenumerate(self.cards):
			if card != None:
				card.do_begin_phase()

	def do_attack_phase(self):
		for _, card in np.ndenumerate(self.cards):
			if card != None:
				card.do_attack_phase()

	def draw(self):
		# Draw the cards in the board
		for x in range(self.size[0]):
			for y in range(self.size[1]):
				card = self.cards[x][y]
				if card != None:
					card_pos = self.grid.get_cell_pos((x,y), align=('center','top'))
					card_pos[0] -= board_card_size[0]//2
					card.draw(card_pos, 'board')

		# (Old) Drawing the mana text number in each cell
		for i, mana in np.ndenumerate(self.red_mana):
			mana_surface = count_font.render(str(mana), True, red)
			self.grid.draw_surface_in_cell(mana_surface, i, align=('right', 'down'), offset=(-2,-2))


		self.grid.draw(grey)

icon_size = 36
icon_padding = 10
def draw_hand(pos):
	for index, item in enumerate(hand):
		pg.draw.circle(screen, blue, (pos[0] + icon_size//2, pos[1] + icon_size//2 + (icon_size+icon_padding)*index), icon_size//2)
		pg.draw.circle(screen, white, (pos[0] + icon_size//2, pos[1] + icon_size//2 + (icon_size+icon_padding)*index), icon_size//2, 2)

# Pygame setup
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (10,50)
pg.init()
screen = pg.display.set_mode((700,800))
card_text_sm = pg.font.Font("Montserrat-Regular.ttf", 18)
card_text_med = pg.font.Font("Montserrat-Regular.ttf", 24)
card_text_lg = pg.font.Font("Montserrat-Regular.ttf", 32)
node_font = pg.font.Font("Montserrat-Regular.ttf", 26)
count_font = pg.font.Font("Montserrat-Regular.ttf", 14)
ui_font = pg.font.Font("Montserrat-Regular.ttf", 24)

# Game setup
game_clock = pg.time.Clock()

game = Game()
# card_pool = CardPool()

potion_card_prototype = Card(name="Potion", cost=1, begin_phase_fns=[lambda self: game.change_health(1, self.owner)])
mountain_card_prototype = Card(name="Mountain", cost=0, passive_fns=[lambda self: board.add_mana(1, 'red', self.pos, 2)])
goblin_card_prototype = CreatureCard(name="Goblin", cost=2, base_power=1, base_health=2)
morale_card_prototype = Card(name="Morale", cost=2, passive_fns=[lambda self: board.buff_creatures_in_range(power=1,health=1,pos=self.pos,distance=2)])

board = Board(grid_count)

hand = Hand()
hand.add_card(potion_card_prototype)
hand.add_card(mountain_card_prototype, 3)
hand.add_card(goblin_card_prototype, 2)
hand.add_card(morale_card_prototype)

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			sys.exit()
		elif event.type == pg.MOUSEBUTTONDOWN:
			if event.button == 1:
				hand.mouse_press(event.pos)
			if event.button == 3:
				board.right_mouse_press(event.pos)
		elif event.type == pg.MOUSEBUTTONUP:
			if event.button == 1:
				hand.mouse_release(event.pos)
		elif event.type == pg.KEYDOWN:
			if event.key == pg.K_ESCAPE:
				sys.exit()
			elif event.key == pg.K_SPACE:
				game.advance_turn()
			elif event.key == pg.K_1:
				hand.add_card(mountain_card_prototype)
			elif event.key == pg.K_2:
				hand.add_card(goblin_card_prototype)
			elif event.key == pg.K_3:
				hand.add_card(morale_card_prototype)
			elif event.key == pg.K_DELETE:
				hand.clear_hand()

	game_clock.tick(60)
	screen.fill(black)

	board.draw()
	hand.draw()
	game.draw()

	pg.display.flip()