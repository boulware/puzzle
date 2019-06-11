import sys
import pygame as pg
import numpy as np
from functools import partial

# General constants
black = (0,0,0)
grey = (127,127,127)
light_grey = (200,200,200)
dark_grey = (60,60,60)
white = (255,255,255)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

# Game parameters
grid_count = (5,5)
node_size = (80,80)
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

	# cells = [start_cell]
	# queue = [{'pos': start_cell, 'd': distance}]

	# if distance > 0:
	# 	while len(queue) > 0:
	# 		start_cell, distance = queue[0]['pos'], queue[0]['d']
	# 		queue.pop(0)
	# 		neighbors = get_neighbors(start_cell)

	# 		for neighbor in neighbors:
	# 			if neighbor not in cells:
	# 				cells.append(neighbor)
	# 				if (distance-1) > 0:
	# 					queue.append({'pos': neighbor, 'd': distance-1})

	# return cells

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
	def get_grid_pos(self, align=('left','up')):
		pos = list(self.rect.topleft)

		if align[0] == 'center':
			pos[0] += self.rect.width//2
		elif align[0] == 'right':
			pos[0] += self.rect.width

		if align[1] == 'center':
			pos[1] += self.rect.height//2
		elif align[1] == 'down':
			pos[1] += self.rect.height	

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
		draw_surface_aligned(source=surface, pos=cell_pos, align=align, offset=offset)

def draw_surface_aligned(source, pos, align=('left','left'), offset=(0,0)):
	new_pos = list(np.add(pos, offset))

	if align[0] == 'center':
		new_pos[0] -= source.get_width()//2
	elif align[0] == 'right':
		new_pos[0] -= source.get_width()

	if align[1] == 'center':
		new_pos[1] -= source.get_height()//2
	elif align[1] == 'down':
		new_pos[1] -= source.get_height()

	screen.blit(source, new_pos)


class ItemPool:
	def __init__(self):
		self.names = ['null']
		self.surfaces = [None]

		self.invalid_surface = node_font.render('?', True, red)

	def add_item(self, name, surface):
		result = self.get_id_by_name(name)
		if result['exists'] == True:
			print("Tried to add item to item pool with duplicate name. Item not added.")
			return {'success': False, 'ID': None}

		self.names.append(name)
		self.surfaces.append(surface)

		return {'success': True, 'ID': len(self.names)-1}

	def item_exists(self, ID):
		if ID < len(self.names) and ID > 0: # Don't include 'null' item as a valid item (ID > 0)
			return True
		else:
			return False

	def get_surface_by_id(self, ID):
		if self.item_exists(ID):
			return self.surfaces[ID]
		else:
			return self.invalid_surface

	def get_id_by_name(self, name):
		for i, item_name in enumerate(self.names):
			if item_name == name and item_name != 'null': # Don't include 'null' item as an item
				return {'exists': True, 'ID': i}

		return {'exists': False, 'ID': None}

	def get_name_by_id(self, ID):
		if self.item_exists(ID):
			return {'exists:': True, 'name': self.names[ID]}
		else:
			print("Tried to reference item with non-existent name or null item.")
			return {'exists': False, 'name': None}

class Inventory:
	def __init__(self):
		self.items = []
		self.count_surfaces = []
		self.grid = Grid(dimensions=(1,0), origin=(10,10), cell_size=node_size)
		self.selected_index = 0

	def add_item(self, name, count=1):
		if count <= 0:
			return

		in_inventory = False
		result = item_pool.get_id_by_name(name)
		if result['exists'] == False:
			print("Tried to add invalid item to inventory.")
			return
		else:
			item_id = result['ID']

		for i, stack in enumerate(self.items):
			if stack['ID'] == item_id:
				in_inventory = True
				stack['count'] += count
				self.update_count_surface(count=stack['count'], index=i)

		if not in_inventory:
			if item_pool.item_exists(item_id):
				self.items.append({'ID': item_id, 'count': count})
				self.update_count_surface(count)
				self.grid.resize((0,1))

	def update_count_surface(self, count, index=None):
		count_surface = count_font.render(str(count), True, white)

		# index=None => add it to the end of the list. i.e., the stack doesn't exist yet
		if index == None:
			self.count_surfaces.append(count_surface)
		else:
			self.count_surfaces[index] = count_surface

	def get_selected_ID(self):
		return self.items[self.selected_index]['ID']

	def click(self):
		result = self.grid.get_cell_at_mouse()
		if result['hit'] == True:
			self.selected_index = result['pos'][1]

		result = board.grid.get_cell_at_mouse()
		if result['hit'] == True:
			board.set_cell(result['pos'], self.get_selected_ID())

	def draw(self):

		result = board.grid.get_cell_at_mouse()
		if result['hit'] == True:
			board.grid.color_cell(result['pos'], black)
			board.grid.draw_surface_in_cell(item_pool.get_surface_by_id(self.get_selected_ID()), result['pos'], ('center','center'))

		result = self.grid.get_cell_at_mouse()
		if result['hit'] == True:
			self.grid.color_cell(result['pos'], dark_grey)

		for i, stack in enumerate(self.items):
			self.grid.draw_surface_in_cell(item_pool.get_surface_by_id(stack['ID']), (0,i), align=('center','center'))
			self.grid.draw_surface_in_cell(self.count_surfaces[i], (0,i), align=('right','down'))

		self.grid.draw(grey)

		if len(self.items) > 0:
			outline_rect = self.grid.get_cell_rect((0,self.selected_index))
			pg.draw.rect(screen, white, outline_rect, 3)

class Game:
	def __init__(self):
		self.ui_font = pg.font.Font("Montserrat-Regular.ttf", 24)

		self.turn_number = 0
		self.__refresh_turn_surface()

		self.player_hp = 20
		self.enemy_hp = 20
		self.__refresh_hp_surfaces()
		self.hp_text_offset = (10,0)

	def __refresh_turn_surface(self):
		self.turn_text= self.ui_font.render("Turn: " + str(self.turn_number), True, white)

	def __refresh_hp_surfaces(self):
		self.player_hp_text = self.ui_font.render("HP: " + str(self.player_hp), True, white)
		self.enemy_hp_text = self.ui_font.render("HP: " + str(self.enemy_hp), True, grey)

	def advance_turn(self):
		self.turn_number += 1
		self.__refresh_turn_surface()

	def draw(self):
		draw_surface_aligned(source=self.turn_text, pos=board.grid.get_grid_pos(align=('left','down')), offset=(0,0))
		draw_surface_aligned(source=self.player_hp_text, pos=board.grid.get_grid_pos(align=('right','down')), align=('left','down'), offset=self.hp_text_offset)
		draw_surface_aligned(source=self.enemy_hp_text, pos=board.grid.get_grid_pos(align=('right','up')), offset=self.hp_text_offset)

class GamePiece:
	def begin_turn(self):


class Board:
	def __init__(self, size):
		self.powers = np.zeros(size, dtype=np.uint8)
		self.item_IDs = np.full(size, 0, dtype=np.uint32)
		self.grid = Grid(size, (100,10), node_size)

	def __reset_powers(self):
		self.powers = np.zeros(self.powers.shape, dtype=np.uint8)

	def set_cell(self, position, item_ID):
		refresh_powers = False
		if self.item_IDs[position] == 2 or item_ID == 2: # If previous or new item is a forest
			refresh_powers = True

		self.item_IDs[position] = item_ID

		if refresh_powers == True:
			self.__reset_powers()
			for i,item_ID in np.ndenumerate(self.item_IDs):
				if item_ID == 2:
					for cell_coord in self.grid.get_cells_by_distance(start_cell=i, distance=1):
						self.powers[cell_coord] += 1

	def begin_turn(self):
		for i, self.item_IDs

	def draw(self):
		for i, item_ID in np.ndenumerate(self.item_IDs):
			if item_ID != 0:
				item_surface = item_pool.get_surface_by_id(item_ID)
				self.grid.draw_surface_in_cell(item_surface, i, align=('center','center'))

		for i, power in np.ndenumerate(self.powers):
			power_surface = count_font.render(str(power), True, green)
			self.grid.draw_surface_in_cell(power_surface, i, align=('right', 'down'), offset=(-2,-2))

		self.grid.draw(grey)

icon_size = 36
icon_padding = 10
def draw_inventory(pos):
	for index, item in enumerate(inventory):
		pg.draw.circle(screen, blue, (pos[0] + icon_size//2, pos[1] + icon_size//2 + (icon_size+icon_padding)*index), icon_size//2)
		pg.draw.circle(screen, white, (pos[0] + icon_size//2, pos[1] + icon_size//2 + (icon_size+icon_padding)*index), icon_size//2, 2)

# Pygame setup
pg.init()
screen = pg.display.set_mode((800,800))
node_font = pg.font.Font("Montserrat-Regular.ttf", 26)
count_font = pg.font.Font("Montserrat-Regular.ttf", 14)

# Game setup
game_clock = pg.time.Clock()

game = Game()
item_pool = ItemPool()

potion_surface = pg.image.load("potion.png")
potion_surface.set_colorkey(white)
result = item_pool.add_item("potion", potion_surface)

forest_surface = pg.Surface(node_size)
pg.draw.circle(forest_surface, green, (node_size[0]//2, node_size[1]//2), 10)
result = item_pool.add_item("forest", forest_surface)

board = Board((7,7))

inventory = Inventory()
inventory.add_item("forest", 3)
inventory.add_item("potion", 5)
# Testing area

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			sys.exit()
		elif event.type == pg.MOUSEBUTTONDOWN:
			if event.button == 1:
				inventory.click()
		elif event.type == pg.KEYDOWN:
			if event.key == pg.K_SPACE:
				game.advance_turn()

	game_clock.tick(60)
	screen.fill(black)

	inventory.draw()
	board.draw()
	game.draw()

	pg.display.flip()