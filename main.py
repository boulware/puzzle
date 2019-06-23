import sys, os, copy, traceback, inspect, socket
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from collections import namedtuple
from typing import NamedTuple, Any
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
gold = (255,215,0)

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
	align_offset = list(offset)

	if align[0] == 'center':
		align_offset[0] -= source.get_width()//2
	elif align[0] == 'right':
		align_offset[0] -= source.get_width()

	if align[1] == 'center':
		align_offset[1] -= source.get_height()//2
	elif align[1] == 'down':
		align_offset[1] -= source.get_height()

	new_pos = list(np.add(pos,align_offset))

	target.blit(source, new_pos)

	return align_offset

hand_card_size = (100,160)
board_card_size = (56,90)

CreatureStats = namedtuple('CreatureStats', 'power max_health')

print_callstack = traceback.print_stack

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
		self.dirty = True

		self._hand_surface = None
		self._board_surface = None

	@property
	def hand_surface(self):
		if not self._hand_surface or self.dirty:
			self.dirty = False
			self.generate_surfaces()

		return self._hand_surface

	@property
	def board_surface(self):
		if not self._board_surface or self.dirty:
			self.dirty = False
			self.generate_surfaces()

		return self._board_surface
	
	

	def _generate_hand_surface(self):
		bg_color = dark_grey
		if self.owner == 0:
			bg_color = dark_red
		elif self.owner == 1:
			bg_color = dark_blue

		self._hand_surface = pg.Surface(hand_card_size)

		pg.draw.rect(self.hand_surface, bg_color, ((0,0), hand_card_size))
		pg.draw.rect(self.hand_surface, light_grey, ((0,0), hand_card_size), 1)
		title_surface = card_text_sm.render(self.name, True, white)
		self.hand_surface.blit(title_surface, (5,0))
		cost_surface = card_text_lg.render(str(self.cost), True, grey)
		draw_surface_aligned(target=self.hand_surface, source=cost_surface, pos=self.hand_surface.get_rect().center, align=('center','center'))

	def _generate_board_surface(self):
		self._board_surface = pg.transform.smoothscale(self.hand_surface, board_card_size)

	@property
	def enemy(self):
		if self.owner == 0:
			return 1
		elif self.owner == 1:
			return 0
		else:
			return None

	def generate_surfaces(self):
		self._generate_hand_surface()
		self._generate_board_surface()

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
		self.buffs = []
		self.dirty = True

	def do_passive(self, field):
		if self.active:
			for fn in self.passive_fns:
				fn(self, field)

	def do_begin_phase(self, field):
		if self.active:
			for fn in self.begin_phase_fns:
				fn(self, field)

	def do_attack_phase(self, field):
		if self.active:
			for fn in self.attack_phase_fns:
				fn(self, field)



	def draw(self, pos, location):
		if location == "hand":
			screen.blit(self.hand_surface, pos)
		if location == "board" or location == "board_hover":
			screen.blit(self.board_surface, pos)

		if self.active == False and location == "board":
			pg.draw.line(screen, red, pos, (pos[0]+board_card_size[0], pos[1]+board_card_size[1]))

class HealthBar:
	def __init__(self, max_health, size):
		self.max_health = max_health
		self.size = size
		self.health = max_health
		self._surface = None
		self.dirty = True

	@property
	def surface(self):
		if not self._surface or self.dirty:
			self.dirty = False
			self._generate_surface()

		return self._surface
	
	def _generate_surface(self):
		self._surface = pg.Surface(self.size)
		red_height = int(self.health/self.max_health*self.size[1])
		pg.draw.rect(self.surface, red, (0, self.size[1]-red_height, self.size[0], self.size[1]))

		pg.draw.line(self.surface, white, (0,0), (0,self.size[1])) # draw left edge
		pg.draw.line(self.surface, white, (self.size[0],0), (self.size[0], self.size[1])) # draw right edge

		# draw borders which delineate cells (max_health+1 because we're drawing borders, not the cells themselves)
		for y in np.linspace(0, self.size[1], self.max_health+1):
			pg.draw.line(self.surface, white, (0,y), (self.size[0],y))

	def _clamp_health(self):
		self.health = np.clip(self.health, 0, self.max_health)

	@property
	def max_health(self):
		return self._max_health
	
	@property
	def max_health(self):
		return self._max_health
	
	@max_health.setter
	def max_health(self, max_health):
		self.dirty = True
		self._max_health = max_health

	@property
	def health(self):
		return self._health

	@health.setter
	def health(self, health):
		self.dirty = True
		self._health = health
	
	def set_health(self, new_health):
		self.dirty = True

		self.health = new_health
		self._clamp_health()

	# def change_health(self, amount):
	# 	if amount == 0:
	# 		return
	# 	self.dirty = True

	# 	self.health += amount
	# 	self._clamp_health()

class CreatureCard(Card):
	def __init__(self, name, cost, base_power, base_max_health, begin_phase_fns=[], attack_phase_fns=[], passive_fns=[]):
		Card.__init__(self=self, name=name, cost=cost, begin_phase_fns=begin_phase_fns, attack_phase_fns=attack_phase_fns)

		self.health_bar = HealthBar(base_max_health, (15,100))

		self._base_power = base_power
		self._base_max_health = base_max_health
		self.health = base_max_health

	@property
	def power(self):
		power = self._base_power
		for buff in self.buffs:
			power += buff[0]

		return power

	@property
	def max_health(self):
		max_health = self._base_max_health
		for buff in self.buffs:
			max_health += buff[1]

		return max_health

	@property
	def health(self):
		return self._health

	@health.setter
	def health(self, health):
		self.dirty = True
		self._health = health
		self._clamp_health()

		self.health_bar.set_health(self.health)

	def change_health(self, amount):
		self.health += amount

	def _clamp_health(self):
		self._health = np.clip(self.health, 0, self.max_health)

	def _generate_hand_surface(self):
		Card._generate_hand_surface(self)

		# Draw power value
		power_text = card_text_sm.render(str(self.power), True, green)
		bottomleft = self.hand_surface.get_rect().bottomleft
		draw_surface_aligned(	target=self.hand_surface, 
								source=power_text,
								pos=bottomleft,
								align=('left','down'),
								offset=(6,-4))

		# Draw health bar
		draw_surface_aligned(	target=self.hand_surface,
								source=self.health_bar.surface,
								pos=hand_card_size,
								align=('right','down'),
								offset=(-1,-1))

		health_text = card_text_med.render(str(self.health), True, red)
		draw_surface_aligned(	target=self.hand_surface,
								source=health_text,
								pos=hand_card_size,
								align=('right','down'),
								offset=(-20,1))


	def _generate_board_surface(self):
		Card._generate_board_surface(self)

	def generate_surfaces(self):
		self._generate_hand_surface()
		self._generate_board_surface()

	def apply_buff(self, power=0, max_health=0):
		buff = (power,max_health)
		if buff != (0,0):
			self.dirty = True
			self.buffs.append(buff)
			self.health_bar.max_health = self.max_health

	def clear_buffs(self):
		Card.clear_buffs(self)
		self.health_bar.max_health = self.max_health
		# if board.check_if_card_is_front(self.pos) == True:
		# 	game.change_health(-self.power, self.enemy)

	def clone(self):
		return CreatureCard(name = self.name,
							cost = self.cost,
							base_power = self._base_power,
							base_max_health = self._base_max_health,
							begin_phase_fns = copy.deepcopy(self.begin_phase_fns),
							attack_phase_fns = copy.deepcopy(self.attack_phase_fns),
							passive_fns = copy.deepcopy(self.passive_fns))

class CardPool:
	def __init__(self):
		self.cards = []

	def add_card(self, card):
		if card in self.cards:
			print("Tried to add card to card pool with same name as one already in pool. Card not added.")
			return {'success': False}

		self.cards.append(card)
		return {'success': True}

	def get_card_by_name(self, name):
		for card in self.cards:
			if card.name == name:
				return card

class Hand:
	def __init__(self, field):
		self.field = field

		self.cards = []

		self.origin = Vec(10,620)
		self.card_spacing = 110

		self.drag_card = None
		self.card_grab_point = None

	def add_card(self, name, count=1):
		card = game.card_pool.get_card_by_name(name)
		if card:
			for _ in range(count):
				self.cards.append(card.clone())
		else:
			print_stack()
			print("Tried to add non-existent card to hand.")

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

			result = self.field.board.grid.get_cell_at_mouse()
			if result['hit'] == True: # If the mouse is hovering over somewhere on the board grid while dragging a card
				pos = result['pos']
				if self.field.board.cards[pos] == None:
					placed_in_board = self.field.board.place_card(result['pos'], self.drag_card)
			
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

			result = self.field.board.grid.get_cell_at_mouse()
			if result['hit'] == True: # If the mouse is hovering over somewhere on the board grid while dragging a card
				pos = result['pos']
				if self.field.board.cards[pos] == None:
					cell_top_left = self.field.board.grid.get_cell_pos(result['pos'], align=('center','top'))
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

# class GameState(Enum):
# 	MainMenu = auto()
# 	Field = auto()

class InputMap:
	def __init__(self, actions):
		self.actions = actions
	def map(self, input):
		self.actions[input]

Input = namedtuple('Input', 'key mouse_button type', defaults=(None,None,'press'))

class TextEntry:
	def __init__(	self,
					pos, font,
					align=('left','top'), text_align=None,
					width=None,
					type='ip',
					label = '',
					text_cursor_scale=0.75, cursor_blink_time=750,
					padding=(5,0),
					default_text='123456789'):
		self.pos = pos
		self.width = width
		self.type = type
		self.align = align
		self.text_align = text_align
		self.font = font
		self.text_cursor_scale = text_cursor_scale
		self.padding = padding
		self.text = default_text
		self.label = label

		if self.width == None:
			if self.type == 'ip':
				self.width = self.font.size('000.000.000.000')[0] + self.padding[0]*2
			else:
				self.width = 200

		self.height = self._calculate_height()
		self.size = (self.width, self.height)
		self.rect = pg.Rect(pos,self.size)

		self.selected_text_indices = None
		self.select_mode = False
		self.select_start_index = None

		self.char_positions = []
		self._calculate_char_positions()

		self.selected = True
		self.cursor_pos = 0
		self.cursor_blink_time = cursor_blink_time
		self.cursor_timer = 0
		self.cursor_visible = True

		self._generate_surfaces()

	def _calculate_char_positions(self, pos=None):
		char_positions = []
		if pos == None: # Recalculate positions for the whole string
			for i in range(0, len(self.text)+1):
				sub_string = self.text[0:i]
				sub_string_width, _ = self.font.size(sub_string)
				char_positions.append(sub_string_width)
		elif pos >= 0:
			char_positions = self.char_positions[:pos]
			for i in range(pos, len(self.text)+1):
				sub_string = self.text[0:i]
				sub_string_width, _ = self.font.size(sub_string)
				char_positions.append(sub_string_width)

		self.char_positions = char_positions # char_positions[n] is the position of the leftmost pixel of the nth character in the text
		# TODO: Only update the ones past pos for center_positions
		# positon_bounds[n] gives the range for when a click inside the textbox should place the cursor in the nth position (this range is between)
		self.position_bounds = []
		current_pos = 0
		for consecutive_positions in zip(self.char_positions[:], self.char_positions[1:]):
			char_center = (consecutive_positions[0]+consecutive_positions[1])//2 # Finds the center of the nth character
			self.position_bounds.append((current_pos, char_center))
			current_pos = char_center + 1

		self.position_bounds.append((current_pos, self.rect.width))

		# print(self.char_positions)
		# print(list(zip(self.char_positions[:], self.char_positions[1:])))

	def _calculate_height(self):
		test_string = ''
		for i in range(32,127):
			test_string += chr(i) # String containing all 'printable' ASCII characters (that we care about)

		return self.font.size(test_string)[1]

	def _generate_surfaces(self):
		self._generate_box_surface()
		self._generate_label_surface()
		self._generate_text_surface()

	def _generate_box_surface(self):
		self.box_surface = pg.Surface(self.size)

		pg.draw.rect(self.box_surface, dark_grey, ((0,0),self.size))
		pg.draw.rect(self.box_surface, white, ((0,0),self.size), 1)

	def _generate_label_surface(self):
		self.label_surface = self.font.render(self.label, True, grey)

	def _generate_text_surface(self):
		self.text_surface = self.font.render(self.text, True, light_grey)

	@property
	def cursor_pos(self):
		return self._cursor_pos

	@cursor_pos.setter
	def cursor_pos(self, cursor_pos):
		if cursor_pos < 0:
			cursor_pos = 0
		elif cursor_pos > len(self.text):
			cursor_pos = len(self.text)

		self._cursor_pos = cursor_pos
		self.cursor_visible = True
		self.cursor_timer = 0
	
	# unselect text and place cursor at cursor_pos
	def _unselect(self, cursor_pos):
		self.cursor_pos = cursor_pos
		self.select_mode = False
		self.selected_text_indices = None
		self.select_start_index = None

	def delete_selected(self):
		left = self.text[:self.selected_text_indices[0]] # left side of selected text
		right = self.text[self.selected_text_indices[1]:] # right ..
		self.text = left + right
		self._unselect(cursor_pos = self.selected_text_indices[0])

	def keypress(self, key, mod, unicode_key):
		if key in range(32,127): # a normal 'printable' character
			if self.selected_text_indices != None:
				self.cursor_pos = self.selected_text_indices[0]
				self.delete_selected()

			self.text = self.text[:self.cursor_pos] + unicode_key + self.text[self.cursor_pos:]
			self.cursor_pos += 1
			self._calculate_char_positions(pos = self.cursor_pos-1)
			self._generate_text_surface()

		if key == pg.K_LEFT:
			if self.cursor_pos == 0:
				pass
			elif mod == pg.KMOD_LSHIFT:
				if self.selected_text_indices == None:
					self.selected_text_indices = (self.cursor_pos-1, self.cursor_pos)
				else:
					if self.cursor_pos == self.selected_text_indices[0]:
						self.selected_text_indices = (self.selected_text_indices[0]-1, self.selected_text_indices[1])
					elif self.cursor_pos == self.selected_text_indices[1]:
						self.selected_text_indices = (self.selected_text_indices[0], self.selected_text_indices[1]-1)
					else:
						print("cursor_pos is not equal to either selected_text_index. something went wrong.")

					if self.selected_text_indices[0] == self.selected_text_indices[1]:
						self.selected_text_indices = None
					else:
						self.selected_text_indices = sorted(self.selected_text_indices)

				self.cursor_pos -= 1
			elif self.selected_text_indices != None:
				self._unselect(cursor_pos=self.selected_text_indices[0])
			else:
				self.cursor_pos -= 1
		elif key == pg.K_RIGHT:
			if self.cursor_pos == len(self.text):
				pass
			elif mod == pg.KMOD_LSHIFT:
				if self.selected_text_indices == None:
					self.selected_text_indices = (self.cursor_pos, self.cursor_pos+1)
				else:
					if self.cursor_pos == self.selected_text_indices[0]:
						self.selected_text_indices = (self.selected_text_indices[0]+1, self.selected_text_indices[1])
					elif self.cursor_pos == self.selected_text_indices[1]:
						self.selected_text_indices = (self.selected_text_indices[0], self.selected_text_indices[1]+1)
					else:
						print("cursor_pos is not equal to either selected_text_index. something went wrong.")

					if self.selected_text_indices[0] == self.selected_text_indices[1]:
						self.selected_text_indices = None
					else:
						self.selected_text_indices = sorted(self.selected_text_indices)

				self.cursor_pos += 1
			elif self.selected_text_indices != None:
				self._unselect(cursor_pos=self.selected_text_indices[0])
			else:
				self.cursor_pos += 1
		elif key == pg.K_BACKSPACE:
			if self.selected_text_indices != None:
				self.delete_selected()
			elif self.cursor_pos > 0:
				self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
				self.cursor_pos -= 1

			self._calculate_char_positions(pos=self.cursor_pos)
			self._generate_text_surface()
		elif key == pg.K_DELETE:
			if self.selected_text_indices != None:
				self.delete_selected()
			elif self.cursor_pos < len(self.text):
				self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]

			self._calculate_char_positions(pos=self.cursor_pos)
			self._generate_text_surface()

	# Returns where the cursor should be placed for the given mouse position
	def mouse_pos_to_cursor_index(self, mouse_pos):
		# mouse position relative to the left side of the textbox
		relative_x = mouse_pos[0] - self.rect.left - self.padding[0]

		for i, position_bound in enumerate(self.position_bounds):
			#print('i=%d; position_bound=%s; mouse_pos=%s; relative_x=%s`'%(i, position_bound, mouse_pos, relative_x))
			if i == 0: # valid between -inf up to the second position_bound
				if relative_x <= position_bound[1]:
					return i
			elif i == len(self.position_bounds)-1: # valid between first position bound and +inf
				if relative_x >= position_bound[0]:
					return i
			elif relative_x >= position_bound[0] and relative_x <= position_bound[1]:
				return i

	def left_mouse_pressed(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			self.selected = True

			self.cursor_pos = self.mouse_pos_to_cursor_index(mouse_pos)

			self.select_start_index = self.cursor_pos
			self.select_mode = True
			self.cursor_visible = True
			self.cursor_timer = 0
		else:
			self.selected = False


	def left_mouse_released(self, mouse_pos):
		self.select_mode = False
		self.select_start_index = None

	def check_mouse_inside(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			return True
		else:
			return False

	def update(self, dt, mouse_pos):
		if self.selected:
			self.cursor_timer += dt
			if self.cursor_timer >= self.cursor_blink_time:
				self.cursor_timer -= self.cursor_blink_time
				self.cursor_visible = not self.cursor_visible

			if self.select_mode == True:
				mouse_index = self.mouse_pos_to_cursor_index(mouse_pos)
				self.cursor_pos = mouse_index
				if self.select_start_index != mouse_index:
					self.selected_text_indices = tuple(sorted([mouse_index, self.select_start_index]))
				else:
					self.selected_text_indices = None


	def _draw_cursor(self):
		if self.cursor_visible:
			x = self.rect.left + self.padding[0] + self.char_positions[self.cursor_pos]
			y_padding = self.rect.height*(1 - self.text_cursor_scale)//2
			pg.draw.line(screen, white, (x,self.rect.top+y_padding), (x,self.rect.bottom-y_padding))

	def _draw_text(self):
		# Ignores self.text_align for now
		screen.blit(self.text_surface, (self.rect.left+self.padding[0], self.rect.top+self.padding[1]))
		if self.selected_text_indices != None:
			left_index = self.selected_text_indices[0]
			right_index = self.selected_text_indices[1]
			left = self.char_positions[left_index]
			right = self.char_positions[right_index]
			shifted_left = left + self.rect.left + self.padding[0]
			shifted_right = right + self.rect.left + self.padding[0]

			pg.draw.rect(screen, light_grey, ((shifted_left,self.rect.top),(shifted_right-shifted_left,self.rect.height)))
			selected_text = self.text[left_index:right_index]
			self.selected_text_surface = self.font.render(selected_text, True, black)
			screen.blit(self.selected_text_surface, (shifted_left, self.rect.top))

	def draw(self):
		draw_surface_aligned(	target=screen,
								source=self.box_surface,
								pos=self.pos,
								align=self.align)

		draw_surface_aligned(	target=screen,
								source=self.label_surface,
								pos=self.pos,
								align=('left','down'))

		self._draw_text()

		if self.selected:
			self._draw_cursor()

class ListMenu:
	def __init__(self, items, pos, align, text_align, font, selected_font, item_spacing=4, selected=0):
		self.items = items
		self.pos = pos
		self.align = align
		self.text_align = text_align
		self.font = font
		self.selected_font = selected_font
		self.item_spacing = item_spacing
		self.selected = selected

	def _generate_surface(self):
		# rects which delineate each menu item (for checking mouse hover, etc.)
		self.item_rects = []

		total_height = 0
		max_width = 0
		for i, item in enumerate(self.items):
			if i == self.selected:
				text_size = self.selected_font.size(item)
			else:
				text_size = self.font.size(item)

			total_height += text_size[1]
			max_width = max(max_width, text_size[0])

		total_height += line_spacing*(len(self.items)-1)

		self.surface = pg.Surface((max_width, total_height))
		# TODO: Only functions with align=center (I think)
		self.rect = pg.Rect((self.pos[0]-max_width//2, self.pos[1]-total_height//2),(max_width,total_height))

		current_y = 0 # The y value where the next menu item should be drawn
		for i, item in enumerate(self.items):
			if i == self.selected:
				text_surface = self.selected_font.render(item, True, gold)
			else:
				text_surface = self.font.render(item, True, light_grey)
			item_pos = (0,0)
			if self.text_align == 'center':
				item_pos = (self.surface.get_width()//2,current_y)
			elif self.text_align == 'left':
				item_pos = (0,current_y)
			elif self.text_align == 'right':
				item_pos = (self.surface.get_width()-1, current_y)

			align_offset = draw_surface_aligned(	target=self.surface, 
													source=text_surface,
													pos=item_pos,
													align=(self.text_align,'top'))

			# Not align-friendly (only works with center)
			item_rect = pg.Rect((self.pos[0]-self.surface.get_width()//2, self.pos[1]-self.surface.get_height()//2+current_y-self.item_spacing//2), (self.surface.get_width(), text_surface.get_height()+self.item_spacing))
			self.item_rects.append(item_rect)

			current_y += text_surface.get_height() + self.item_spacing

	@property
	def selected(self):
		return self._selected

	@selected.setter
	def selected(self, selected):
		self._selected = selected
		self._generate_surface()

	def move_cursor_up(self):
		self.selected -= 1
		if self.selected < 0:
			self.selected = len(self.items)-1

	def move_cursor_down(self):
		self.selected += 1
		if self.selected >= len(self.items):
			self.selected = 0

	def get_selected_item(self):
		return self.items[self.selected]

	# Menu items can be selected but not hovered. Sometimes, when clicking,
	# you may not want to activate the item unless it's still being hovered
	# (i.e., the mouse is still over the menu element)
	def check_mouse_inside(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			return True
		else:
			return False


	def update(self, dt, mouse_pos):
		# TODO: Can probably be optimized by doing one check for the WHOLE self.surface first and then continuing if True
		for i, item_rect in enumerate(self.item_rects):
			if item_rect.collidepoint(mouse_pos):
				self.selected = i
				return

	def draw(self):
		draw_surface_aligned(	target=screen,
								source=self.surface,
								pos=self.pos,
								align=self.align)
		# for item_rect in self.item_rects:
		# 	pg.draw.rect(screen, green, item_rect, 1)


class GameState:
	def handle_input(self, input, mouse_pos, mod=None, unicode_key=None):
		state = None
		if input in self.input_map:
			state = self.input_map[input](mouse_pos)

		if state != None:
			return state 	# if we have a state change, go ahead and return,
		else:				# but if not, let's check the 'any key' event:
			if input.key:
				state = self.input_map[Input(key='any')](input.key, mod, unicode_key)
	def update(self, dt, mouse_pos):
		raise NotImplementedError()
	# These are 'virtual' methods -- should be overridden by child class
	def draw(self):
		raise NotImplementedError()

class MainMenu(GameState):
	def __init__(self):
		self.input_map = {
			Input(key='any'): lambda key, mod, unicode_key: self.any_key_pressed(key, mod, unicode_key),
			Input(key=pg.K_SPACE): lambda _: self.keyboard_select_menu_item(),
			Input(key=pg.K_RETURN): lambda _: self.keyboard_select_menu_item(),
			Input(mouse_button=1): lambda mouse_pos: self.left_mouse_press(mouse_pos),
			Input(key=pg.K_ESCAPE): lambda _: sys.exit(),
			Input(key=pg.K_UP): lambda _: self.list_menu.move_cursor_up(),
			Input(key=pg.K_w): lambda _: self.list_menu.move_cursor_up(),
			Input(key=pg.K_LEFT): lambda _: self.list_menu.move_cursor_up(),
			Input(key=pg.K_a): lambda _: self.list_menu.move_cursor_up(),
			Input(key=pg.K_DOWN): lambda _: self.list_menu.move_cursor_down(),
			Input(key=pg.K_s): lambda _: self.list_menu.move_cursor_down(),
			Input(key=pg.K_RIGHT): lambda _: self.list_menu.move_cursor_down(),
			Input(key=pg.K_d): lambda _: self.list_menu.move_cursor_down()
		}


		self.list_menu = ListMenu(	items=('Play', 'Connect', 'Exit'),
									pos=(screen_size[0]//2, screen_size[1]//2),
									align=('center','center'),
									text_align=('center'),
									font=main_menu_font,
									selected_font=main_menu_selected_font)

	def update(self, dt, mouse_pos):
		self.list_menu.update(dt, mouse_pos)

	def draw(self):
		self.list_menu.draw()

	def any_key_pressed(self, key, mod, unicode_key):
		pass

	def activate_menu(self):
		selected = self.list_menu.get_selected_item()
		if selected == 'Play':
			return Field()
		elif selected == 'Connect':
			return ConnectMenu()
		elif selected == 'Exit':
			sys.exit()

	def left_mouse_press(self, mouse_pos):
		if self.list_menu.check_mouse_inside(mouse_pos):
			return self.activate_menu()

	def keyboard_select_menu_item(self):
		return self.activate_menu()

class ConnectMenu(GameState):
	def __init__(self):
		self.input_map = {
			Input(key='any'): lambda key, mod, unicode_key: self.any_key_pressed(key, mod, unicode_key),
			Input(key=pg.K_ESCAPE): lambda _: self.cancel(),
			Input(mouse_button=1): lambda mouse_pos: self.left_mouse_pressed(mouse_pos),
			Input(mouse_button=1, type='release'): lambda mouse_pos: self.left_mouse_released(mouse_pos),
			Input(key=pg.K_RETURN): lambda _: self.submit()
		}

		self.text_entry = TextEntry(pos=(screen_size[0]//2-100,screen_size[1]//2),
									type='ip',
									font=main_menu_font_med,
									label='IP Address')

	def submit(self):
		print("(fake) Attempting to connect to %s" % self.text_entry.text)

	def any_key_pressed(self, key, mod, unicode_key):
		self.text_entry.keypress(key, mod, unicode_key)

	def left_mouse_pressed(self, mouse_pos):
		self.text_entry.left_mouse_pressed(mouse_pos)

	def left_mouse_released(self, mouse_pos):
		self.text_entry.left_mouse_released(mouse_pos)

	def cancel(self):
		return MainMenu()

	def update(self, dt, mouse_pos):
		self.text_entry.update(dt, mouse_pos)

	def draw(self):
		self.text_entry.draw()

class Field(GameState):
	def __init__(self):
		self.board = Board(self, grid_count)
		self.hand = Hand(self)
		self.hand.add_card("Potion")
		self.hand.add_card("Mountain", 3)
		self.hand.add_card("Goblin", 2)
		self.hand.add_card("Morale")

		self.input_map = {
			Input(key='any'): lambda key, mod, unicode_key: self.any_key_pressed(key, mod, unicode_key),
			Input(mouse_button=1): lambda mouse_pos: self.hand.mouse_press(mouse_pos),
			Input(mouse_button=1, type='release'): lambda mouse_pos: self.hand.mouse_release(mouse_pos),
			Input(mouse_button=3): lambda mouse_pos: self.board.right_mouse_press(mouse_pos),
			Input(key=pg.K_SPACE): lambda mouse_pos: game.advance_turn(),
			Input(key=pg.K_1): lambda mouse_pos: self.hand.add_card("Mountain"),
			Input(key=pg.K_2): lambda mouse_pos: self.hand.add_card("Goblin"),
			Input(key=pg.K_3): lambda mouse_pos: self.hand.add_card("Morale"),
			Input(key=pg.K_DELETE): lambda mouse_pos: self.hand.clear_hand(),
			Input(key=pg.K_ESCAPE): lambda mouse_pos: self.go_to_main_menu(),
		}

	def any_key_pressed(self, key, mod, unicode_key):
		pass

	def go_to_main_menu(self):
		return MainMenu()

	def update(self, dt, mouse_pos):
		pass

	def draw(self):
		self.board.draw()
		self.hand.draw()

class Game:
	def __init__(self):
		self.card_pool = CardPool()

		potion_card_prototype = Card(name="Potion", cost=1, begin_phase_fns=[lambda self, field: game.change_health(1, self.owner)])
		mountain_card_prototype = Card(name="Mountain", cost=0, passive_fns=[lambda self, field: field.board.add_mana(1, 'red', self.pos, 1)])
		goblin_card_prototype = CreatureCard(name="Goblin", cost=2, base_power=1, base_max_health=2)
		morale_card_prototype = Card(name="Morale", cost=2, passive_fns=[lambda self, field: field.board.buff_creatures_in_range(power=1,max_health=1,pos=self.pos,distance=2)])

		self.card_pool.add_card(potion_card_prototype)
		self.card_pool.add_card(mountain_card_prototype)
		self.card_pool.add_card(goblin_card_prototype)
		self.card_pool.add_card(morale_card_prototype)


		self.turn_display = TurnDisplay()

		self._turn_number = 0
		self.__start_turn()
		self.__refresh_turn_surface()

		self.player_healths = [20,20]
		self.__refresh_hp_surfaces()
		self.hp_text_offset = (10,0)

		self.state = MainMenu()

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

	@property
	def board(self):
		if isinstance(self.state, Field):
			return self.state.board

	# TODO: Move this to Field (probably)
	def advance_turn(self):
		if self.board:
			if self._phase_name == "Begin":
				self.board.do_begin_phase()
				self.__advance_phase()
			elif self._phase_name == "Attack":
				self.board.do_attack_phase()
				self.__advance_phase()
			elif self._phase_name == "End":
				self._turn_number += 1
				self.__start_turn()

			self.__refresh_turn_surface()

	def handle_input(self, input, mouse_pos, mod=None, unicode_key=None):
		state = self.state.handle_input(input, mouse_pos, mod, unicode_key)
		if state:
			self.state = state

	def update(self, dt, mouse_pos):
		self.state.update(dt, mouse_pos)

	def draw(self):
		self.state.draw()

		if isinstance(self.state, Field):
			self.turn_display.draw(self.state.board.grid.get_grid_pos(align=('right','center'),offset=(50,0)), align=('left','center'))
			draw_surface_aligned(target=screen, source=self._bottom_hp_text, pos=self.state.board.grid.get_grid_pos(align=('right','down')), align=('left','down'), offset=self.hp_text_offset)
			draw_surface_aligned(target=screen, source=self._top_hp_text, pos=self.state.board.grid.get_grid_pos(align=('right','up')), offset=self.hp_text_offset)


class Board:
	def __init__(self, field, size):
		self.field = field

		self.size = size
		self.cards = np.full(size, None, np.dtype(Card))
		self.grid = Grid(dimensions=size, origin=(10,10), cell_size=node_size)
		self._reset_mana()

	def place_card(self, cell, card):
		if self.grid.check_cell_valid(cell) == True:
			card.pos = cell
			card.owner = self.get_cell_owner(cell)
			self.cards[cell] = card
			self._refresh_passives()

			return True # Successfully fulfilled requirements for placing the card and placed it.
		else:
			print("Tried to place card in invalid cell")
			return False

	def return_card_to_hand(self, cell):
		if self.cards[cell] != None:
			self.cards[cell].owner = None
			self.field.hand.add_card(name=self.cards[cell].name)
			self.cards[cell] = None
			self._refresh_passives()

			return True # Card returned
		else:
			return False # No card returned

	def remove_card_from_board(self, cell):
		if self.cards[cell] != None:
			self.cards[cell].owner = None
			self.cards[cell] = None
			self._refresh_passives()

	def right_mouse_press(self, pos):
		result = self.grid.get_cell_at_mouse()
		if result['hit'] == True:
			cell = result['pos']
			self.return_card_to_hand(cell)
			self._refresh_passives()

	def get_cell_owner(self, cell):
		if self.grid.check_cell_valid(cell) == True:
			if cell[1] < grid_count[1]//2:
				return 1
			else:
				return 0
		else:
			print("Tried to get cell owner of invalid cell")
			return None

	def get_frontmost_occupied_cell(self, player, lane):
		ranks = []
		if player == 0:
			ranks = range(3,6,1) #3,4,5
		elif player == 1:
			ranks = range(2,-1,-1) #2,1,0
		else:
			print("get_frontmost_occupied_cell got invalid player")
			return {'error:': True}

		for rank in ranks:
			if self.cards[lane, rank] != None:
				return {'error': False,
						'cell': (lane, rank)}

		return {'error': False,
				'cell': None} # There are no cards in the lane



	# def check_if_card_is_front(self, cell):
	# 	if self.grid.check_cell_valid(cell) == False:
	# 		print("check_if_card_is_front() got invalid cell")
	# 		return {'error': True}

	# 	if self.get_frontmost_occupied_cell(
	# 		return {'error': False,
	# 				'result': True}

	def _reset_mana(self):
		self.red_mana = np.zeros(self.size, dtype=np.uint8)

	def _refresh_passives(self):
		dirty = False

		for cell, card in np.ndenumerate(self.cards):
			if card != None:
				if self.red_mana[cell] >= card.cost and card.active != True:
					dirty = True
					card.active = True
				elif self.red_mana[cell] < card.cost and card.active != False:
					dirty = True
					card.active = False

		self._reset_mana()
		self._clear_buffs()
		self.do_passive()

		if dirty == True:
			self._refresh_passives() # Iterative refreshes when state has changed
			# This is necessary because of the complex interactions between cards

	def _clear_buffs(self):
		for _, card in np.ndenumerate(self.cards):
			if card != None:
				card.clear_buffs()

	def add_mana(self, amount, type, pos, distance=1):
		if pos:
			if type == "red":
				cell_coords = self.grid.get_cells_by_distance(pos, distance)
				for cell_coord in cell_coords:
					self.red_mana[cell_coord] += amount

	def buff_creatures_in_range(self, power, max_health, pos, distance=1):
		if pos:
			cell_coords = self.grid.get_cells_by_distance(pos, distance)
			for cell_coord in cell_coords:
				if isinstance(self.cards[cell_coord], CreatureCard):
					self.cards[cell_coord].apply_buff(power=1,max_health=1)

	def do_passive(self):
		#traceback.print_stack()
		for _, card in np.ndenumerate(self.cards):
			if card != None:
				card.do_passive(self.field)

	def do_begin_phase(self):
		for _, card in np.ndenumerate(self.cards):
			if card != None:
				card.do_begin_phase(self.field)

	def do_attack_phase(self):
		for _, card in np.ndenumerate(self.cards):
			if card != None:
				card.do_attack_phase(self.field)

		for lane in range(self.size[0]): # 0,1,...,4
			front0_cell = self.get_frontmost_occupied_cell(0, lane)['cell']
			front1_cell = self.get_frontmost_occupied_cell(1, lane)['cell']

			card_0 = None
			card_1 = None

			if front0_cell:
				card_0 = self.cards[front0_cell]
			if front1_cell:
				card_1 = self.cards[front1_cell]

			is_creature_0 = isinstance(card_0, CreatureCard)
			is_creature_1 = isinstance(card_1, CreatureCard)

			if is_creature_0 and is_creature_1:
				if card_0.active:
					card_1.change_health(-card_0.power)
				if card_1.active:
					card_0.change_health(-card_1.power)

				if card_0.health <= 0:
					self.remove_card_from_board(front0_cell)
				if card_1.health <= 0:
					self.remove_card_from_board(front1_cell)

			if is_creature_0 and not is_creature_1:
				if card_0.active:
					if card_1:
						self.remove_card_from_board(front1_cell)
					else:
						game.change_health(-card_0.power, 1)
			if is_creature_1 and not is_creature_0:
				if card_1.active:
					if card_0:
						self.remove_card_from_board(front0_cell)
					else:
						game.change_health(-card_1.power, 0)

			if not is_creature_0 and not is_creature_1:
				pass

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

# Represents the maps between an input and an action
#Control = namedtuple('Control', 'input action')

# Pygame setup
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (10,50)
pg.init()
pg.key.set_repeat(300, 40)
screen_size = (700,800)
screen = pg.display.set_mode(screen_size)
card_text_sm = pg.font.Font("Montserrat-Regular.ttf", 18)
card_text_med = pg.font.Font("Montserrat-Regular.ttf", 24)
card_text_lg = pg.font.Font("Montserrat-Regular.ttf", 32)
node_font = pg.font.Font("Montserrat-Regular.ttf", 26)
count_font = pg.font.Font("Montserrat-Regular.ttf", 14)
ui_font = pg.font.Font("Montserrat-Regular.ttf", 24)
main_menu_font = pg.font.Font("Montserrat-Regular.ttf", 48)
main_menu_font_med = pg.font.Font("Montserrat-Regular.ttf", 32)
main_menu_selected_font = pg.font.Font("Montserrat-Regular.ttf", 60)

# Game setup
game_clock = pg.time.Clock()

input = Input()
game = Game()

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			sys.exit()
		elif event.type == pg.MOUSEBUTTONDOWN:
			input = Input(mouse_button=event.button, type='press')
			game.handle_input(input=input, mouse_pos=event.pos)
		elif event.type == pg.MOUSEBUTTONUP:
			input = Input(mouse_button=event.button, type='release')	
			game.handle_input(input=input, mouse_pos=event.pos)
		elif event.type == pg.KEYDOWN:
			input = Input(key=event.key, type='press')
			game.handle_input(input=input, mouse_pos=pg.mouse.get_pos(), mod=event.mod, unicode_key=event.unicode)

	# Update
	dt = game_clock.tick(60)

	game.update(dt, pg.mouse.get_pos())

	# Draw
	screen.fill(black)

	game.draw()

	pg.display.flip()