import sys, os, copy, traceback, inspect, socket, selectors, types
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
dark_grey = (40,40,40)
white = (255,255,255)
red = (255,0,0)
light_red = (255,100,100)
dark_red = (70,0,0)
very_dark_red = (40,0,0)
green = (0,255,0)
light_green = (0,150,0)
dark_green = (0,70,0)
very_dark_green = (0,40,0)
blue = (0,0,255)
light_blue = (100,100,255)
dark_blue = (0,0,70)
very_dark_blue = (0,0,40)
gold = (255,215,0)
pink = (255,200,200)

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
		self._generate_surface()

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
		self._generate_surface()

		return True

	# Returns whether the cell is on 'player 0's (return 0) side of the board or 'player 1's (return 1)
	def get_cell_owner(self, cell):
		if cell[1] >= 0 and cell[1] <= 2:
			return 1
		if cell [1] >= 3 and cell[1] <= 5:
			return 0

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

		return {'hit': hit, 'cell': (grid_x, grid_y)}

	def _generate_surface(self):
		self.surface = pg.Surface((self.rect.size[0]+1, self.rect.size[1]+1))
		pg.draw.rect(self.surface, very_dark_blue, ((0,0), (self.rect.width, self.rect.height//2)))
		pg.draw.rect(self.surface, very_dark_red, ((0,self.rect.height//2),(self.rect.width,self.rect.height//2)))

		grid_color = white

		for x in range(self.dimensions[0]+1):
			x_start = x*self.cell_size[0]
			pg.draw.line(self.surface, grid_color, (x_start, 0), (x_start, self.cell_size[1]*self.dimensions[1]))
		for y in range(self.dimensions[1]+1):
			y_start = y*self.cell_size[1]
			pg.draw.line(self.surface, grid_color, (0, y_start), (self.cell_size[0]*self.dimensions[0], y_start))		

	def draw(self, color=white, player_perspective=0):
		if self.drawable:
			if player_perspective == 0:
				to_flip = False
			elif player_perspective == 1:
				to_flip = True
			screen.blit(pg.transform.flip(self.surface, False, to_flip), self.rect.topleft)

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

def draw_surface_aligned(target, source, pos, align=('left','left'), offset=(0,0), alpha=255):
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
card_proportion = hand_card_size[0]/hand_card_size[1]
board_card_size = (int(node_size[0]*card_proportion),node_size[1])

print_callstack = traceback.print_stack

class Card:
	def __init__(self, name, cost, begin_phase_fns=[], attack_phase_fns=[], passive_fns=[]):
		self.name = name
		self.cost = cost
		self.cell = None
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
		# if self.owner == 0:
		# 	bg_color = dark_red
		# elif self.owner == 1:
		# 	bg_color = dark_blue

		self._hand_surface = pg.Surface(hand_card_size)

		pg.draw.rect(self.hand_surface, bg_color, ((0,0), hand_card_size))
		pg.draw.rect(self.hand_surface, light_grey, ((0,0), hand_card_size), 1)
		title_surface = card_text_sm.render(self.name, True, white)
		self.hand_surface.blit(title_surface, (5,0))
		cost_surface = card_text_lg.render(str(self.cost), True, light_grey)
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

	def draw(self, pos, location, hover=False):
		if location == "hand":
			screen.blit(self.hand_surface, pos)
			if hover:
				pg.draw.rect(screen, gold, (pos, self.hand_surface.get_size()), 3)
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
		# if board.check_if_card_is_front(self.cell) == True:
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
				return card.clone()

class Hand:
	def __init__(self, field):
		self.cards = []
		self.field = field

	def __iter__(self):
		return iter(self.cards)

	def __getitem__(self, key):
		if key < 0 or key >= self.card_count:
			raise LookupError('Invalid hand index')

		return self.cards[key]
		
	@property
	def card_count(self):
		return len(self.cards)

	def add_card(self, name, count=1):
		card = self.field.game.card_pool.get_card_by_name(name)
		if card:
			for _ in range(count):
				self.cards.append(card.clone())
		else:
			print_stack()
			print("Tried to add non-existent card to hand.")

	def pop_card(self, index):
		return self.cards.pop(index)

	def clear_hand(self):
		self.cards = []

class Phase:
	def __init__(self, phase_names=[], initial_phase_ID=0):
		self.names = phase_names
		self.ID = initial_phase_ID
		self.turn_ended = False
	
	def __iter__(self):
		return iter(self.names)

	@property
	def ID(self):
		return self._ID

	@ID.setter
	def ID(self, new_ID):
		self._ID = new_ID

		if new_ID >= len(self.names):
			self.turn_ended = True
		else:
			self._name = self.names[new_ID]

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, new_name):
		for ID, name in enumerate(self.names):
			if new_name == name:
				self.ID = ID # name is autoset by @ID.setter

	# Return True if advanced past the last phase,
	# Return False otherwise
	def advance_phase(self):
		if not self.turn_ended:
			self.ID += 1

			if self.ID >= len(self.names):
				return True
			else:
				return False
		else:
			return True

	def end_turn(self):
		self.turn_ended = False
		self.ID = 0

line_spacing = 9

class TurnDisplay:
	def __init__(self, phase, font):
		self.font = font
		self.phase = phase

	@property
	def phase(self):
		return self._phase

	@phase.setter
	def phase(self, new_phase):
		self._phase = new_phase
		self._generate_phase_texts()
	
	def _generate_phase_texts(self):
		self.phase_texts = []
		self.phase_active_texts = []

		for name in self.phase.names:
			self.phase_texts.append(self.font.render(name, True, white))
			self.phase_active_texts.append(self.font.render(name, True, green))

	def draw(self, pos):
		line_spacing = self.font.get_linesize()
		for phase_ID, _ in enumerate(self.phase):
			if phase_ID == self.phase.ID:
				text_set = self.phase_active_texts # Draw from the active text set
			else:
				text_set = self.phase_texts # Draw from non-active text set

			screen.blit(text_set[phase_ID], (pos[0],pos[1]+line_spacing*phase_ID))


class InputMap:
	def __init__(self, actions):
		self.actions = actions
	def map(self, input):
		self.actions[input]

Input = namedtuple('Input', 'key mouse_button type', defaults=(None,None,'press'))

class UI_Element:
	def __init__(self, parent_container=None):
		self.parent_container = parent_container
		self.events = []
	def get_event(self):
		if len(self.events) > 0:
			return self.events.pop(0)
	def any_key_pressed(self, key, mod, unicode_key):
		pass
	def left_mouse_pressed(self, mouse_pos):
		pass
	def left_mouse_released(self, mouse_pos):
		pass
	def update(self, dt, mouse_pos):
		pass

class UI_Container:
	def __init__(self):
		self.ui_elements = []
		self.ui_group_parent = None
		self.focused_element = None

	def __iter__(self):
		return iter(self.ui_elements)

	def add_ui_element(self, element):
		self.ui_elements.append(element)
		element.parent_container = self
		if self.focused_element == None:
			self.focused_element = element

	def focus_ui_element(self, target):
		for e in self.ui_elements:
			if e == target:
				self.focused_element = target
				if self.ui_group_parent:
					self.ui_group_parent.focus_ui_element(target)
				return True
		return False
		# If the element isn't in this container, ignore the focus request

	# Returns True if there is no focused element upon return,
	# Returns False if there is still a focused element upon return
	def unfocus_ui_element(self, target=None):
		if target == None:
			self.focused_element = None
		else:
			if target == self.focused_element:
				self.focused_element = None

		for e in self.ui_elements:
			if isinstance(e, UI_Container):
				e.unfocus_ui_element(target)

	def any_key_pressed(self, key, mod, unicode_key):
		if self.focused_element:
			self.focused_element.any_key_pressed(key, mod, unicode_key)

	def left_mouse_pressed(self, mouse_pos):
		for e in self.ui_elements:
			e.left_mouse_pressed(mouse_pos)

	def left_mouse_released(self, mouse_pos):
		if self.focused_element:
			self.focused_element.left_mouse_released(mouse_pos)

	def update(self, dt, mouse_pos):
		for e in self.ui_elements:
			e.update(dt, mouse_pos)

	def draw(self):
		for e in self.ui_elements:
			e.draw()
			if e == self.focused_element:
				pass#pg.draw.circle(screen, pink, e.pos, 10)

# Represents a group of individual UI_Containers that are displayed
# on the screen at once and interact smoothly between each other
class UI_Group:
	def __init__(self, ui_containers):
		self.ui_containers = ui_containers
		self.focused_container = None
		for container in self.ui_containers:
			container.unfocus_ui_element()
			container.ui_group_parent = self

	def __iter__(self):
		ui_elements = []
		for container in self.ui_containers:
			for e in container:
				ui_elements.append(e)

		return iter(ui_elements)

	def focus_ui_element(self, target):
		for container in self.ui_containers:
			for e in container:
				if e == target:
					container.focused_element = target
					self.focused_container = container
		for container in self.ui_containers:
			if container != self.focused_container:
				container.unfocus_ui_element()

	def unfocus_ui_element(self, target=None):
		for container in self.ui_containers:
			container.unfocus_ui_element(target)
		self.focused_container = None


	def any_key_pressed(self, key, mod, unicode_key):
		for container in self.ui_containers:
			container.any_key_pressed(key, mod, unicode_key)

	def left_mouse_pressed(self, mouse_pos):
		for container in self.ui_containers:
			container.left_mouse_pressed(mouse_pos)

	def left_mouse_released(self, mouse_pos):
		for container in self.ui_containers:
			container.left_mouse_released(mouse_pos)

	def update(self, dt, mouse_pos):
		for container in self.ui_containers:
			container.update(dt, mouse_pos)

	def draw(self):
		for container in self.ui_containers:
			container.draw()

class Label(UI_Element):
	def __init__(self, pos, font, align=('left','top'), text_color=white, text='', parent_container=None):
		UI_Element.__init__(self, parent_container)
		self.pos = pos
		self.font = font
		self.align = align
		self.text_color = text_color
		self.text = text

		self._generate_surface()

	@property
	def text(self):
		return self._text

	@text.setter
	def text(self, new_text):
		self._text = new_text
		self._generate_surface()
	

	def _generate_surface(self):
		self.surface = self.font.render(self.text, True, self.text_color)

	def draw(self):
		draw_surface_aligned(target=screen, source=self.surface, pos=self.pos, align=self.align)


class Button(UI_Element):
	def __init__(	self,
					pos, font,
					text,
					align=('left','top'),
					bg_colors={'default': black, 'hovered': dark_grey, 'pressed': green},
					text_colors={'default': white, 'hovered': white, 'pressed': white},
					padding=(10,0),
					parent_container=None):
		UI_Element.__init__(self, parent_container)
		self.pos = pos
		self.font = font
		self.text = text
		self.padding = padding
		self.align = align
		self.bg_colors = bg_colors
		self.text_colors = text_colors

		self.width = self.font.size(text)[0] + self.padding[0]*2
		self.height = self.font.size(text)[1] + self.padding[1]*2

		self._hovered = False
		self._pressed = False
		self.button_was_pressed = False

		self._generate_surfaces()

	@property
	def size(self):
		return (self.width, self.height)

	@property
	def rect(self):
		offset_x, offset_y = 0, 0
		if self.align[0] == 'center':
			offset_x -= self.width//2
		elif self.align[0] == 'right':
			offset_x -= self.width

		if self.align[1] == 'center':
			offset_y -= self.height//2
		elif self.align[1] == 'down':
			offset_y -= self.height

		return pg.Rect((self.pos[0]+offset_x, self.pos[1]+offset_y), self.size)

	def _generate_surfaces(self):
		self.surfaces = {	'default': pg.Surface(self.size),
							'hovered': pg.Surface(self.size),
							'pressed': pg.Surface(self.size)}

		for key, surface in self.surfaces.items():
			pg.draw.rect(surface, self.bg_colors[key], ((0,0),self.size))
			pg.draw.rect(surface, white, ((0,0),self.size),1)
			surface.blit(self.font.render(self.text, True, self.text_colors[key]), self.padding)

	@property
	def hovered(self):
		return self._hovered

	@hovered.setter
	def hovered(self, hovered):
		self._hovered = hovered

	@property
	def pressed(self):
		return self._pressed

	@pressed.setter
	def pressed(self, pressed):
		self._pressed = pressed
	
	def left_mouse_pressed(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			self.pressed = True
			if self.parent_container:
				self.parent_container.focus_ui_element(self)

	def left_mouse_released(self, mouse_pos):
		if self.pressed == True and self.rect.collidepoint(mouse_pos):
			self.button_was_pressed = True
		self.pressed = False

	def clear_pressed(self):
		self.button_was_pressed = False

	def update(self, dt, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			self.hovered = True
		else:
			self.hovered = False
			self.pressed = False

	@property
	def state(self):
		_state = 'default'
		if self.pressed:
			_state = 'pressed'
		elif self.hovered:
			_state = 'hovered'

		return _state

	def draw(self):
		draw_offset = draw_surface_aligned(	target=screen,
											source=self.surfaces[self.state],
											pos=self.pos,
											align=self.align)

class TextEntry(UI_Element):
	def __init__(	self,
					pos, font,
					align=('left','top'), text_align=None,
					width=None,
					type='ip',
					label = '',
					text_cursor_scale=0.75, cursor_blink_time=750,
					padding=(5,0),
					alpha=255,
					default_text='',
					parent_container=None):
		UI_Element.__init__(self, parent_container)
		self.pos = pos
		self.width = width
		self.type = type
		self.align = align
		self.text_align = text_align
		self.font = font
		self.alpha = alpha
		self.text_cursor_scale = text_cursor_scale
		self.padding = padding
		self.text = default_text
		self.label = label

		if self.width == None:
			if self.type == 'ip':
				self.width = self.font.size('000.000.000.000')[0] + self.padding[0]*2
			elif self.type == 'port':
				self.width = self.font.size('00000')[0] + self.padding[0]*2
			else:
				self.width = 200

		self.size = (self.width, self.height)
		self.rect = pg.Rect(pos,self.size)

		self.selected_text_indices = None
		self.select_mode = False
		self.select_start_index = None

		self.char_positions = []
		self._calculate_char_positions()

		self.selected = False
		self.cursor_pos = 0
		self.cursor_blink_time = cursor_blink_time
		self.cursor_timer = 0
		self.cursor_visible = True

		self._generate_surfaces()

	@property
	def height(self):
		return self._calculate_height()
	
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

		self.box_surface.set_alpha(self.alpha)

	def _generate_label_surface(self):
		self.label_surface = self.font.render(self.label, True, grey)
		self.label_surface.set_alpha(self.alpha)

	def _generate_text_surface(self):
		self.text_surface = self.font.render(self.text, True, light_grey)
		self.text_selected_surface = self.font.render(self.text, True, black)

		self.text_surface.set_alpha(self.alpha)
		self.text_selected_surface.set_alpha(self.alpha)

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
		
	def clear_text(self):
		self.text = ''
		self.selected_text_indices = None
		self.select_mode = False
		self.select_start_index = None
		self.cursor_pos = 0
		self.cursor_timer = 0
		self.cursor_visible = True

		self._calculate_char_positions()
		self._generate_surfaces()

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

	def any_key_pressed(self, key, mod, unicode_key):
		if self.selected == False:
			return

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
				self._unselect(cursor_pos=self.selected_text_indices[1])
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
			if i == len(self.position_bounds)-1: # valid between first position bound and +inf
				if relative_x >= position_bound[0]:
					return i
			elif relative_x >= position_bound[0] and relative_x <= position_bound[1]:
				return i

		print('mouse_pos_to_cursor_index() failed')
		return 0

	def left_mouse_pressed(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			if self.parent_container:
				self.parent_container.focus_ui_element(self)
			self.selected = True

			new_cursor_pos = self.mouse_pos_to_cursor_index(mouse_pos)
			if new_cursor_pos:
				self.cursor_pos = new_cursor_pos

			self.select_start_index = self.cursor_pos
			self.select_mode = True
			self.cursor_visible = True
			self.cursor_timer = 0
		else:
			if self.parent_container:
				self.parent_container.unfocus_ui_element(self)
			self.selected = False


	def left_mouse_released(self, mouse_pos):
		self.select_mode = False
		self.select_start_index = None


		# if self.selected_text_indices[0] == self.selected_text_indices[1]:
		# 	self.selected_text_indices = None

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

			pg.draw.rect(screen, grey, ((shifted_left,self.rect.top),(shifted_right-shifted_left,self.rect.height)))
			screen.blit(self.text_selected_surface, (shifted_left, self.rect.top), (left, 0, right-left, self.text_selected_surface.get_height()))

	def draw(self):
		draw_surface_aligned(	target=screen,
								source=self.box_surface,
								pos=self.pos,
								align=self.align,
								alpha=self.alpha)

		draw_surface_aligned(	target=screen,
								source=self.label_surface,
								pos=self.pos,
								align=('left','down'),
								alpha=self.alpha)

		self._draw_text()

		if self.selected:
			self._draw_cursor()

class ListMenu(UI_Element):
	def __init__(self, items, pos, align, text_align, font, selected_font, item_spacing=4, selected=0, parent_container=None):
		UI_Element.__init__(self, parent_container)
		self.items = items
		self.pos = pos
		self.align = align
		self.text_align = text_align
		self.font = font
		self.selected_font = selected_font
		self.item_spacing = item_spacing
		self.selected = selected
		self.confirmed_index = None

	def _generate_surfaces(self):
		self.item_surfaces = []
		self.selected_item_surfaces = []

		for item in self.items:
			self.item_surfaces.append(self.font.render(item, True, light_grey))
			self.selected_item_surfaces.append(self.selected_font.render(item, True, gold))

	@property
	def rect(self):
		current_height = 0
		max_width = 0
		for item_index, _ in enumerate(self.items):
			item_surface = self.get_item_surface(item_index)
			current_height += item_surface.get_height()
			max_width = max(max_width, item_surface.get_width())

		return pg.Rect(self.pos, (max_width, current_height))

	@property
	def confirmed_item_text(self):
		if self.confirmed_index != None:
			return self.items[self.confirmed_index]
		else:
			return ''
	

	@property
	def selected(self):
		return self._selected

	@selected.setter
	def selected(self, selected):
		self._selected = selected
		self._generate_surfaces()

	def clear_confirmed(self):
		self.confirmed_index = None

	def _move_cursor_up(self):
		self.selected -= 1
		if self.selected < 0:
			self.selected = len(self.items)-1

	def _move_cursor_down(self):
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

	def get_item_surface(self, item_index):
		if item_index == self.selected:
			return self.selected_item_surfaces[item_index]
		else:
			return self.item_surfaces[item_index]

	def any_key_pressed(self, key, mod, unicode_key):
		if key==pg.K_UP or key==pg.K_w or key==pg.K_LEFT or key==pg.K_a:
			self._move_cursor_up()
		elif key==pg.K_DOWN or key==pg.K_s or key==pg.K_RIGHT or key==pg.K_d:
			self._move_cursor_down()
		elif key==pg.K_RETURN or key==pg.K_SPACE:
			self.confirmed_index = self.selected

	def get_hovered_item(self, mouse_pos):
		current_y = 0
		mouse_relative_pos = (mouse_pos[0] - self.pos[0], mouse_pos[1] - self.pos[1])
		for item_index, _ in enumerate(self.items):
			item_surface = self.get_item_surface(item_index)
			# x bounds are already satisfied because of the self.rect.collidepoint() check above
			if mouse_relative_pos[1] >= current_y and mouse_relative_pos[1] < current_y+item_surface.get_height(): # y bounds
				return item_index
			current_y += item_surface.get_height()

	def left_mouse_pressed(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			self.parent_container.focus_ui_element(self)

			hovered = self.get_hovered_item(mouse_pos)
			if hovered != None:
				self.selected = hovered

			self.confirmed_index = self.selected


	def update(self, dt, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			self.parent_container.focus_ui_element(self)

			hovered = self.get_hovered_item(mouse_pos)
			if hovered != None:
				self.selected = hovered


	def draw(self):
		current_y = 0
		for item_index, _ in enumerate(self.items):
			item_surface = self.get_item_surface(item_index)
			screen.blit(item_surface, (self.pos[0], self.pos[0]+current_y))
			current_y += item_surface.get_height()

		# draw_surface_aligned(	target=screen,
		# 						source=self.surface,
		# 						pos=self.pos,
		# 						align=self.align)
		# for item_rect in self.item_rects:
		# 	pg.draw.rect(screen, green, item_rect, 1)


class GameState:
	def __init__(self, game):
		self.game = game
		self.ui_container = UI_Container()
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


		self.list_menu = ListMenu(	items=('Start SP Game', 'Start MP Game', 'Host', 'Connect', 'Exit'),
									pos=(300,300),
									align=('center','center'),
									text_align=('center'),
									font=main_menu_font,
									selected_font=main_menu_selected_font)

		self.ui_container.add_ui_element(self.list_menu)

	def enter(self):
		self.ui_container.focus_ui_element(self.list_menu)

	def update(self, dt, mouse_pos):
		if self.list_menu.confirmed_index != None:
			selected_text = self.list_menu.confirmed_item_text
			if selected_text == 'Start SP Game':
				self.queue_state_transition(Field(self.game, 0))
			elif selected_text == 'Start MP Game':
				if self.game.connection_role == 'host':
					self.queue_state_transition(Field(self.game, 0))
				elif self.game.connection_role == 'client':
					self.queue_state_transition(Field(self.game, 1))
			elif selected_text == 'Host':
				self.queue_state_transition(HostMenu(self.game))
			elif selected_text == 'Connect':
				self.queue_state_transition(ConnectMenu(self.game))
			elif selected_text == 'Exit':
				sys.exit()

		self.list_menu.clear_confirmed()

	def draw(self):
		pass#UI_Container.draw(self)

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

		self.port_textentry = TextEntry(pos=(screen_size[0]//2-100,screen_size[1]//2+100),
										type='port',
										font=main_menu_font_med,
										label='Port',
										default_text='4141')

		self.host_button = Button(	pos=(screen_size[0]//2-100,screen_size[1]//2+200),
									font=main_menu_font_med,
									text='Host')

		self.disconnect_button = Button(pos=(screen_size[0]//2-100,screen_size[1]//2+250),
										font=main_menu_font_med,
										text='Disconnect')

		self.ui_container.add_ui_element(self.port_textentry)
		self.ui_container.add_ui_element(self.host_button)
		self.ui_container.add_ui_element(self.disconnect_button)

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

		self.ip_textentry = TextEntry(	pos=(screen_size[0]//2-100,screen_size[1]//2),
										type='ip',
										font=main_menu_font_med,
										label='IP Address',
										default_text='localhost')

		self.port_textentry = TextEntry(pos=(screen_size[0]//2-100,screen_size[1]//2+100),
										type='port',
										font=main_menu_font_med,
										label='Port',
										default_text='4141')

		self.connect_button = Button(	pos=(screen_size[0]//2-100,screen_size[1]//2+200),
										font=main_menu_font_med,
										text='Connect')

		self.ui_container.add_ui_element(self.ip_textentry)
		self.ui_container.add_ui_element(self.port_textentry)
		self.ui_container.add_ui_element(self.connect_button)

		self.sel = None


	def _attempt_to_connect(self, host, port):
		self.game._attempt_to_connect(host, port)

	def return_to_menu(self):
		self.queue_state_transition(MainMenu(self.game))

	def update(self, dt, mouse_pos):
		if self.connect_button.button_was_pressed == True:
			self._attempt_to_connect(self.ip_textentry.text, int(self.port_textentry.text))
			self.connect_button.clear_pressed()

def clamp(value, min_value, max_value):
	clamped_value = value
	if value < min_value:
		clamped_value = min_value
	elif value > max_value:
		clamped_value = max_value

	return clamped_value

# Scales color towards (0,0,0), where amount is between 0 and 1 (1 taking it all the way to black)
def darken_color(color, amount):
	try:
		new_color = list(color)
		for i, channel in enumerate(new_color):
			new_color[i] = clamp(channel*(1-amount), 0, 255)
		return new_color
	except:
		print("Failed to darken color.")
		return color

# Scales color towards (0,0,0), where amount is between 0 and 1 (1 takes it all the way to white)
def lighten_color(color, amount):
	try:
		new_color = list(color)
		for i, channel in enumerate(new_color):
			new_color[i] = clamp(channel + amount*(255-channel), 0, 255)
		return new_color
	except:
		print("Failed to darken color.")
		return color

class Field(GameState):
	def __init__(self, game, player_number, game_type='SP'):
		GameState.__init__(self, game)

		self.player_number = player_number
		self.game_type = game_type
		self.board = Board(self, grid_count)
		self.hands = {
			0: Hand(self), # Hand for player 0
			1: Hand(self) # ..	 ..  ..     1
		}
		for _, hand in self.hands.items():
			hand.add_card("Potion")
			hand.add_card("Mountain", 3)
			hand.add_card("Goblin", 2)
			hand.add_card("Morale")

		self.player_turn = 0 # Player 'id' of the player whose turn it is.
		self.cards_played = 0 # Tracks number of cards played on current turn

		self.player_healths = [20,20]
		self.hand_area_height = 80


		self.phase = Phase(['End'])
		self.turn_display = TurnDisplay(self.phase, ui_font)

		self.hand_origin = Vec(self.board.grid.get_grid_pos(align=('left','down'),offset=(0,50)))
		self.hand_center = Vec(screen_size[0]//2, screen_size[1]-100)
		self.hand_spacing = Vec(110,0)
		self.drag_card = None
		self.card_grab_point = None

		self.turn_button = Button(	pos=self.board.grid.get_grid_pos(align=('left','down'),offset=(-100,-50)),
									align=('right','up'),
									font=ui_font,
									text="End Turn",
									parent_container=self)

		if player_number == 0:
			self.player_health_labels = [
			Label(	pos=self.board.grid.get_grid_pos(align=('right','down'), offset=(10,-30)),
					font=ui_font,
					text='Player 0: %s'%self.player_healths[0]),
			Label(	pos=self.board.grid.get_grid_pos(align=('right','up'), offset=(10,0)),
					font=ui_font,
					text='Player 1: %s'%self.player_healths[1])
			]
		elif player_number == 1:
			self.player_health_labels = [
			Label(	pos=self.board.grid.get_grid_pos(align=('right','up'), offset=(10,0)),
					font=ui_font,
					text='Player 0: %s'%self.player_healths[0]),
			Label(	pos=self.board.grid.get_grid_pos(align=('right','down'), offset=(10,-50)),
					font=ui_font,
					text='Player 1: %s'%self.player_healths[1])
			]

		self.ui_container.add_ui_element(self.turn_button)
		for label in self.player_health_labels:
			self.ui_container.add_ui_element(label)


		self.input_map = {
			# Input(key='any'): lambda key, mod, unicode_key: self.any_key_pressed(key, mod, unicode_key),
			Input(mouse_button=1): lambda mouse_pos: self.left_mouse_pressed(mouse_pos),
			Input(mouse_button=1, type='release'): lambda mouse_pos: self.left_mouse_released(mouse_pos),
			Input(mouse_button=3): lambda mouse_pos: self.right_mouse_press(mouse_pos),
			Input(key=pg.K_SPACE): lambda mouse_pos: self.space_pressed(),
			Input(key=pg.K_1): lambda mouse_pos: self.hands[self.player_turn].add_card("Mountain"),
			Input(key=pg.K_2): lambda mouse_pos: self.hands[self.player_turn].add_card("Goblin"),
			Input(key=pg.K_3): lambda mouse_pos: self.hands[self.player_turn].add_card("Morale"),
			Input(key=pg.K_DELETE): lambda mouse_pos: self.hands[self.player_turn].clear_hand(),
			Input(key=pg.K_ESCAPE): lambda mouse_pos: self.go_to_main_menu()
		}

	def change_health(self, amount, player):
		if player == 0 or player == 1:
			self.player_healths[player] += amount
			self.player_health_labels[player].text = 'Player %d: %d'%(player, self.player_healths[player])
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
		return pg.Rect((hand_left, self.hand_center[1]), (self.active_hand.card_count*self.hand_spacing[0], hand_card_size[1]))
	
	def space_pressed(self):
		if self.game_type == 'MP' and not self.is_current_player_active():
			return
		self._advance_turn()


	def left_mouse_pressed(self, mouse_pos):
		if self.hand_rect.collidepoint(mouse_pos): # mouse is hovering hand
			hand_left = self.get_left_side_of_hand()
			relative_x = int((mouse_pos[0] - hand_left) % self.hand_spacing[0])
			relative_y = int(mouse_pos[1] - self.hand_center[1])
			clicked_card_index = int((mouse_pos[0] - hand_left) // self.hand_spacing[0])

			if relative_x >= 0 and relative_x < hand_card_size[1]:
				if clicked_card_index >= 0 and clicked_card_index < self.active_hand.card_count:
					self.drag_card = self.active_hand.pop_card(clicked_card_index)
					self.card_grab_point = (relative_x,relative_y)

	def is_current_player_active(self):
		if self.player_turn == self.player_number:
			return True
		else:
			return False

	def left_mouse_released(self, mouse_pos):
		if self.drag_card:
			placed_in_board = False # True if card is placed onto the board during this mouse release

			if self.is_current_player_active():
				result = self.board.grid.get_cell_at_mouse()
				if result['hit'] == True: # If the mouse is hovering over somewhere on the board grid while dragging a card
					if self.cards_played < 1:
						if self.player_number == 0:
							pos = result['cell']
						elif self.player_number == 1:
							pos = (result['cell'][0],self.board.size[1]-1-result['cell'][1])
						if self.board.cards[pos] == None and self.board.grid.get_cell_owner(pos) == self.player_number:
							placed_in_board = self.board.place_card(pos, self.drag_card)
							send_string = 'card placed;' + self.drag_card.name + ';' + str(pos[0]) + ';' + str(pos[1]) + ';[END]'
							self.game.queue_network_data(send_string.encode('utf-8'))
							self.cards_played += 1
			
			if placed_in_board == False:
				self.active_hand.add_card(name=self.drag_card.name)
			
			self.drag_card = None
			self.card_grab_point = None # Probably not necessary

		if self.turn_button.left_mouse_released(mouse_pos): # if button was pressed
			if self.game_type == 'SP':
				self._end_turn()
			elif self.game_type == 'MP':
				if self.is_current_player_active():
					self._end_turn()
	
	def right_mouse_press(self, mouse_pos):
		self.board.right_mouse_press(mouse_pos)

	def set_active_player(self, player_number):
		self.player_turn = player_number
		self._end_turn()

	def swap_active_player(self):
		if self.player_turn == 0:
			self.player_turn = 1
		elif self.player_turn == 1:
			self.player_turn = 0

	def _end_turn(self):
		end_of_turn = self._advance_turn()
		while end_of_turn == False:
			end_of_turn = self._advance_turn()

	def _advance_turn(self):
		if self.phase.name == "End":
			self.board.do_begin_phase()
			self.board.do_attack_phase()

		if self.is_current_player_active():
			send_string = 'phase advanced' + ';[END]'
			self.game.queue_network_data(send_string.encode('utf-8'))

		self.phase.advance_phase()

		if self.phase.turn_ended == True:
			self.phase.end_turn()
			self.swap_active_player()
			self.cards_played = 0
			return True
		else:
			return False

	def process_network_data(self, data):
		raw_data_string = data.decode('utf-8')
		event_strings = raw_data_string.split('[END]')
		for event_string in event_strings:
			args = event_string.split(';')
			try:
				if args[0] == 'card placed':
					self.board.place_card((int(args[2]),int(args[3])),self.game.card_pool.get_card_by_name(args[1]))
				elif args[0] == 'turn ended':
					print('turn ended')
					self._end_turn()
				elif args[0] == 'phase advanced':
					self._advance_turn()
				elif args[0] == 'health changed':
					self.change_health(int(args[1]), int(args[2]))
				elif args[0] == 'message sent':
					self.chat_window.add_message(args[1], args[2])
			except:
				pass

	def go_to_main_menu(self):
		send_string = 'quit field' + ';[END]'
		self.game.queue_network_data(send_string.encode('utf-8'))
		self.queue_state_transition(MainMenu(self.game))

	def generate_hand_card_positions(self):
		total_width = self.active_hand.card_count * self.hand_spacing[0]
		return [Vec(self.hand_center[0]+i*self.hand_spacing[0]-total_width//2,self.hand_center[1]) for i in range(self.active_hand.card_count)]

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
				card_right = card_pos[0] + hand_card_size[0]

				if mouse_pos[0] >= card_left and mouse_pos[0] < card_right:
					self.hovered_card_index = i
					return

		self.hovered_card_index = None

	def update(self, dt, mouse_pos):
		self._generate_hovered_card_index(mouse_pos)
		# self.turn_button.update(dt, mouse_pos)

	def draw(self):
		if self.player_number == 0:
			current_player_color = red
			other_player_color = blue
		elif self.player_number == 1:
			current_player_color = blue
			other_player_color = red

		if self.player_turn == 0:
			active_player_color = red
		elif self.player_turn == 1:
			active_player_color = blue

		# Draw board
		self.board.draw(player_perspective=self.player_number)

		# Calculate colors for card areas based on which player is active
		if self.is_current_player_active():
			my_card_area_color = darken_color(current_player_color,0.65)
			other_card_area_color = dark_grey
		else:
			my_card_area_color = dark_grey
			other_card_area_color = darken_color(other_player_color,0.65)

		# Draw my card area
		pg.draw.rect(	screen, my_card_area_color,
						((0,screen_size[1]-self.hand_area_height),
						(screen_size[0], self.hand_area_height))
					)
		pg.draw.line(	screen, lighten_color(my_card_area_color,0.5),
						(0,screen_size[1]-self.hand_area_height),
						(screen_size[0],screen_size[1]-self.hand_area_height)
					)
		# Draw other card area
		pg.draw.rect(	screen, other_card_area_color,
						((0,0), (screen_size[0], self.hand_area_height)))
		pg.draw.line(	screen, lighten_color(other_card_area_color,0.5),
						(0, self.hand_area_height),
						(screen_size[0], self.hand_area_height))


		# Draw cards in hand
		for i, card_pos in enumerate(self.generate_hand_card_positions()):
			if i == self.hovered_card_index:
				hover = True
			else:
				hover = False

			self.active_hand[i].draw(pos=card_pos, location='hand', hover=(i==self.hovered_card_index))

		# Draw active player text and circle
		active_player_text = "Player %d"%self.player_turn
		text_h_padding = 10
		text_size = ui_font.size(active_player_text)
		padded_size = (text_size[0]+2*text_h_padding, text_size[1])
		active_player_text_surface = pg.Surface(padded_size)
		pg.draw.rect(active_player_text_surface, white, ((0,0),(padded_size)))
		active_player_text_surface.blit(ui_font.render(active_player_text, True, active_player_color), (text_h_padding,0))
		draw_pos = (20, screen_size[1]//2)
		offset = draw_surface_aligned(	target=screen,
										source=active_player_text_surface,
										pos=draw_pos,
										align=('left','down'))
		pg.draw.circle(screen, active_player_color,
						(	int(draw_pos[0] + offset[0] + active_player_text_surface.get_width() + 20),
							int(draw_pos[1] + offset[1] + active_player_text_surface.get_height()//2)),
						15)
		pg.draw.circle(screen, white,
						(	int(draw_pos[0] + offset[0] + active_player_text_surface.get_width() + 20),
							int(draw_pos[1] + offset[1] + active_player_text_surface.get_height()//2)),
						15, 1)

		# Draw turn display
		self.turn_display.draw(pos=self.board.grid.get_grid_pos(align=('right','center'),offset=(50,0)))

		# Draw card being dragged
		if self.drag_card:
			drawn_in_board = False # True if the drag card gets drawn in the board this frame rather than floating on screen

			result = self.board.grid.get_cell_at_mouse()
			if result['hit'] == True: # If the mouse is hovering over somewhere on the board grid while dragging a card
				if self.player_number == 0:
					pos = result['cell']
				elif self.player_number == 1:
					pos = (result['cell'][0],self.board.size[1]-1-result['cell'][1])
				if self.board.cards[pos] == None:
					cell_top_left = self.board.grid.get_cell_pos(result['cell'], align=('center','top'))
					cell_top_left[0] -= board_card_size[0]//2
					self.drag_card.draw(cell_top_left, "board_hover")
					drawn_in_board = True
			
			if drawn_in_board == False:
				mouse_coords = Vec(pg.mouse.get_pos())
				self.drag_card.draw(mouse_coords - self.card_grab_point, "hand")
			

class Game:
	def __init__(self, start_state=None):
		self.card_pool = CardPool()

		potion_card_prototype = Card(name="Potion", cost=1, begin_phase_fns=[lambda self, field: field.change_health(amount=1, player=self.owner)])
		mountain_card_prototype = Card(name="Mountain", cost=0, passive_fns=[lambda self, field: field.board.add_mana(amount=1, type='red', cell=self.cell, distance=1)])
		goblin_card_prototype = CreatureCard(name="Goblin", cost=2, base_power=1, base_max_health=2)
		morale_card_prototype = Card(name="Morale", cost=2, passive_fns=[lambda self, field: field.board.buff_creatures_in_range(power=1,max_health=1,cell=self.cell,distance=2)])

		self.card_pool.add_card(potion_card_prototype)
		self.card_pool.add_card(mountain_card_prototype)
		self.card_pool.add_card(goblin_card_prototype)
		self.card_pool.add_card(morale_card_prototype)

		self.network_data_queue = []

		self.connection_label = Label(	pos=(0,0),
										font=main_menu_font_small,
										text='',
										text_color=green,
										align=('left','up'))

		self.chat_window = ChatWindow(	name_font=chat_name_font,
										message_font=chat_message_font,
										name_width=75,
										message_width=300,
										log_height=150)

		self.ui_container = UI_Container()
		self.ui_container.add_ui_element(self.connection_label)
		self.ui_container.add_ui_element(self.chat_window)

		self.input_map = {
			Input(key='any'): lambda key, mod, unicode_key: self.any_key_pressed(key, mod, unicode_key),
			Input(mouse_button=1): lambda mouse_pos: self.left_mouse_pressed(mouse_pos),
			Input(mouse_button=1, type='release'): lambda mouse_pos: self.left_mouse_released(mouse_pos)
		}

		self.state = start_state(self)

		self.selector = None
		self.socket = None
		self.accepting_connections = False
		self.connected = False
		self.connected_to_address = None
		self.connection_role = None

	def refresh_ui_group(self):
		self.ui_group = UI_Group(ui_containers=[self.ui_container, self.state.ui_container])

	def any_key_pressed(self, key, mod, unicode_key):
		self.ui_group.any_key_pressed(key, mod, unicode_key)

	def left_mouse_pressed(self, mouse_pos):
		self.ui_group.left_mouse_pressed(mouse_pos)

	def left_mouse_released(self, mouse_pos):
		self.ui_group.left_mouse_released(mouse_pos)

	def add_chat_message(self, user, text):
		self.chat_window.add_message(user=user, text=text)

	@property
	def state(self):
		return self._state

	@state.setter
	def state(self, new_state):
		self._state = new_state
		self.refresh_ui_group()
		self._state.enter()
	
	@property
	def connected(self):
		return self._connected
	
	@connected.setter
	def connected(self, new_state):
		if new_state == True:
			self.connection_label.text = "Connected to %s"%str(self.connected_to_address)
		else:
			self.connection_label.text = ''

		self._connected = new_state

	@property
	def accepting_connections(self):
		return self._accepting_connections
	
	@accepting_connections.setter
	def accepting_connections(self, new_state):
		if new_state == True:
			self.connection_label.text = "Accepting Connections"
		else:
			self.connection_label.text = ''
		
		self._accepting_connections = new_state
	

	def start_host(self, port):
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
		self.selector = selectors.DefaultSelector()

		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.setblocking(False)
		self.socket.connect_ex((host,port))
		events = selectors.EVENT_READ | selectors.EVENT_WRITE
		data = types.SimpleNamespace(	connid=0,
										msg_total=1,
										recv_total=0,
										messages=[b'SIMON IS CUTE'],
										outb=b"")

		self.selector.register(self.socket, events, data=data)
		self.connection_role = 'client'
		print('Connected to %s:%s'%(host,port))
		self.connected = True
		self.connection_label.text = 'Connected to %s:%s'%(host, port)

	def _service_connection_as_host(self, key, mask):
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
				sent = sock.send(packet)

			self.network_data_queue = []

	def _service_connection_as_client(self, key, mask):
		sock = key.fileobj
		data = key.data
		if mask & selectors.EVENT_READ:
			recv_data = sock.recv(1024)  # Should be ready to read
			if recv_data:
				print("received", repr(recv_data), "from connection", data.connid)
				self.process_network_data(recv_data)
#				data.recv_total += len(recv_data)
		elif mask & selectors.EVENT_WRITE:
			for packet in self.network_data_queue:
				print("sending", repr(packet), "to connection", data.connid)
				sent = sock.send(packet)  # Should be ready to write

			self.network_data_queue = []

	def close_connection(self):
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
		self.network_data_queue.append(data)

	def process_network_data(self, data):
		print('process_network_data')
		raw_data_string = data.decode('utf-8')
		event_strings = raw_data_string.split('[END]')
		for event_string in event_strings:
			args = event_string.split(';')
			try:
				if args[0] == 'message sent':
					self.chat_window.add_message(args[1], args[2])
			except:
				pass

		self.state.process_network_data(data)

	def is_valid_player(self, player):
		if player == 0 or player == 1:
			return True
		else:
			return False

	@property
	def board(self):
		if isinstance(self.state, Field):
			return self.state.board

	def handle_input(self, input, mouse_pos, mod=None, unicode_key=None):
		if input in self.input_map:
			self.input_map[input](mouse_pos)
		else:
			self.input_map[Input(key='any')](input.key, mod, unicode_key)

		if self.ui_group.focused_container == None or self.ui_group.focused_container.focused_element == None:
			self.state.handle_input(input, mouse_pos, mod, unicode_key)

	def get_player_name(self):
		if self.connection_role == None:
			return 'Offline'
		elif self.connection_role == 'host':
			return 'Tyler'
		elif self.connection_role == 'client':
			return 'Shawn'

	def update(self, dt, mouse_pos):
		if self.state.target_state:
			self.state = self.state.target_state

		self.select()
		self.state.update(dt, mouse_pos)
		self.ui_group.update(dt, mouse_pos)

		for ui_element in self.ui_group:
			event = ui_element.get_event()
			while event != None:
				if event[0] == 'send chat message':
					self.chat_window.add_message(user=self.get_player_name(), text=event[1])
					send_string = 'message sent;' + self.get_player_name() + ';' + event[1] + ';[END]'
					self.queue_network_data(send_string.encode('utf-8'))

				event = ui_element.get_event()


		# if self.ui_group.focused_container:
		# 	if self.ui_group.focused_container.focused_element:
		# 		print(self.ui_group.focused_container.focused_element)
		# else:
		# 	print('**')

	def draw(self):
		self.state.draw()
		self.ui_group.draw()

def split_text(text, font, word_wrap_width):
		lines = []

		split_text = text.split(' ')
		current_line = ''
		for i, word in enumerate(split_text):
			if i == 0:
				line_width = font.size(word)[0]
			else:
				line_width = font.size(current_line + ' ' + word)[0]
			if line_width >= word_wrap_width:
				lines.append(current_line)
				current_line = word
			else:
				if i == 0:
					current_line += word
				else:
					current_line += ' ' + word

		if len(current_line) > 0:
			lines.append(current_line)

		return lines

def draw_text(text, pos, font, color=white, word_wrap=False, word_wrap_width=None):
	if len(text) == 0:
		line_count = 0
	if word_wrap == False:
		text_surface = pg.Surface(font.size(text))
		text_surface.set_colorkey(black)
		text_surface.fill(black)
		text_surface.blit(font.render(text, True, color), (0,0))
		line_count = 1
	else:
		line_spacing = font.get_linesize()
		lines = split_text(text, font, word_wrap_width=word_wrap_width)
		line_count = len(lines)
		text_surface = pg.Surface((word_wrap_width, line_spacing*line_count))
		text_surface.set_colorkey(black)
		text_surface.fill(black)

		for line_number, line in enumerate(lines):
			text_surface.blit(font.render(line, True, color), (0,line_number*line_spacing))

	screen.blit(text_surface, pos)

	return line_count


class ChatWindow(UI_Element):
	def __init__(self, name_font, message_font, name_width, message_width, log_height, text_color=white, parent_container=None):
		UI_Element.__init__(self, parent_container)
		self.pos = (0,0)
		self.name_font = name_font
		self.message_font = message_font
		self.name_width = name_width
		self.message_width = message_width
		self.log_height = log_height
		self.text_color = text_color
		self.user_colors = {"Tyler": light_red, "Shawn": light_blue, "Offline": grey}
		self.messages = []

		self.text_entry = TextEntry(pos=(self.pos[0], self.pos[1]+self.log_height),
									font=message_font,
									type='chat',
									width=message_width+name_width,
									alpha=128)

		self.ui_container = UI_Container()
		self.ui_container.add_ui_element(self.text_entry)

	@property
	def width(self):
		return self.name_width + self.message_width

	@property
	def height(self):
		return self.log_height + self.text_entry.height
	
	@property
	def rect(self):
		return pg.Rect(self.pos, (self.width, self.height))
	
	def add_message(self, user, text):
		message = (user, text)
		print(message)
		self.messages.append(message)

	def any_key_pressed(self, key, mode, unicode_key):
		self.ui_container.any_key_pressed(key, mode, unicode_key)
		if key == pg.K_RETURN:
			if len(self.text_entry.text) > 0:
				self.events.append(('send chat message', self.text_entry.text))
				self.text_entry.clear_text()

	def left_mouse_pressed(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			if self.parent_container:
				self.parent_container.focus_ui_element(self)
		else:
			if self.parent_container:
				self.parent_container.unfocus_ui_element()

		self.ui_container.left_mouse_pressed(mouse_pos)

	def left_mouse_released(self, mouse_pos):
		self.ui_container.left_mouse_released(mouse_pos)

	def update(self, dt, mouse_pos):
		self.ui_container.update(dt, mouse_pos)

	def draw(self):
		background_surface = pg.Surface(self.rect.size)
		background_surface.set_alpha(128)
		background_surface.fill(dark_grey)
		screen.blit(background_surface, self.pos)

		self.ui_container.draw()

		line_spacing = self.message_font.get_linesize() + 4
		current_line_count = 0

		for message in self.messages[::-1]: # Look through messages backwards, since we only show the most recent ones
			this_line_count = len(split_text(text=message[1], font=self.message_font, word_wrap_width=self.message_width))
			current_line_count += this_line_count
			draw_text(	text=message[0],
						pos=(self.pos[0], self.pos[1] + self.log_height - current_line_count*line_spacing),
						font=self.name_font,
						color = self.user_colors[message[0]],
						word_wrap = False)
			draw_text(	text=message[1],
						pos=(self.name_width + self.pos[0], self.pos[1] + self.log_height - current_line_count*line_spacing),
						font = self.message_font,
						color = lighten_color(self.user_colors[message[0]], 0.5),
						word_wrap = True,
						word_wrap_width = self.message_width)




class Board:
	def __init__(self, field, size):
		self.field = field

		self.size = size
		self.cards = np.full(size, None, np.dtype(Card))
		grid_origin = (screen_size[0]//2-int((size[0]*node_size[0])//2), screen_size[1]//2-int((size[1]*node_size[1])//2))
		self.grid = Grid(dimensions=size, origin=grid_origin, cell_size=node_size)
		self._reset_mana()

	def place_card(self, cell, card):
		if self.grid.check_cell_valid(cell) == True:
			card.cell = cell
			card.owner = self.grid.get_cell_owner(cell)
			self.cards[cell] = card
			self._refresh_passives()

			return True # Successfully fulfilled requirements for placing the card and placed it.
		else:
			print("Tried to place card in invalid cell")
			return False

	def return_card_to_hand(self, cell):
		if self.cards[cell] != None:
			self.cards[cell].owner = None
			self.field.active_hand.add_card(name=self.cards[cell].name)
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
			cell = result['cell']
			self.return_card_to_hand(cell)
			self._refresh_passives()

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

	def add_mana(self, amount, type, cell, distance=1):
		if cell:
			if type == "red":
				cell_coords = self.grid.get_cells_by_distance(start_cell=cell, distance=distance)
				for cell_coord in cell_coords:
					self.red_mana[cell_coord] += amount

	def buff_creatures_in_range(self, power, max_health, cell, distance=1):
		if cell:
			cell_coords = self.grid.get_cells_by_distance(start_cell=cell, distance=distance)
			for cell_coord in cell_coords:
				if isinstance(self.cards[cell_coord], CreatureCard):
					self.cards[cell_coord].apply_buff(power=1,max_health=1)

	def do_passive(self):
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
						self.field.change_health(-card_0.power, 1)
			if is_creature_1 and not is_creature_0:
				if card_1.active:
					if card_0:
						self.remove_card_from_board(front0_cell)
					else:
						self.field.change_health(-card_1.power, 0)

			if not is_creature_0 and not is_creature_1:
				pass

	def draw(self, player_perspective=0):
		self.grid.draw(grey, player_perspective=player_perspective)

		# Draw the cards in the board
		for x in range(self.size[0]):
			for y in range(self.size[1]):
				card = self.cards[x][y]
				if card != None:
					if player_perspective == 1:
						y = self.size[1] - 1 - y
					card_pos = self.grid.get_cell_pos((x,y), align=('center','top'))
					card_pos[0] -= board_card_size[0]//2
					card.draw(card_pos, 'board')

		# (Old) Drawing the mana text number in each cell
		for i, mana in np.ndenumerate(self.red_mana):
			mana_surface = count_font.render(str(mana), True, red)
			if self.field.player_number == 0:
				cell = i
			else:
				cell = (i[0],self.grid.dimensions[1]-1-i[1])
			self.grid.draw_surface_in_cell(mana_surface, cell, align=('right', 'down'), offset=(-2,-2))

try:
	if sys.argv[2] == 'l':
		os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (10,50)
	elif sys.argv[2] == 'r':
		os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (700,50)
except:
	pass

# pg setup
pg.init()
pg.key.set_repeat(300, 30)
screen_size = (800,800)
screen = pg.display.set_mode(screen_size)
card_text_sm = pg.font.Font("Montserrat-Regular.ttf", 18)
card_text_med = pg.font.Font("Montserrat-Regular.ttf", 24)
card_text_lg = pg.font.Font("Montserrat-Regular.ttf", 32)
node_font = pg.font.Font("Montserrat-Regular.ttf", 26)
count_font = pg.font.Font("Montserrat-Regular.ttf", 14)
ui_font = pg.font.Font("Montserrat-Regular.ttf", 24)
main_menu_font = pg.font.Font("Montserrat-Regular.ttf", 48)
main_menu_font_med = pg.font.Font("Montserrat-Regular.ttf", 32)
main_menu_font_small = pg.font.Font("Montserrat-Regular.ttf", 18)
chat_message_font = pg.font.Font("Montserrat-Regular.ttf", 16)
chat_name_font = pg.font.Font("Montserrat-Regular.ttf", 16)
main_menu_selected_font = pg.font.Font("Montserrat-Regular.ttf", 60)

# Game setup
game_clock = pg.time.Clock()

input = Input()

start_state = lambda game_: MainMenu(game_)
try:
	if sys.argv[1] == 'field':
		start_state = lambda game_: Field(game_, 0, 'SP')
	elif sys.argv[1] == 'connect':
		start_state = lambda game_: ConnectMenu(game_)
	elif sys.argv[1] == 'host':
		start_state = lambda game_: HostMenu(game_)
except IndexError:
	pass

game = Game(start_state)

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