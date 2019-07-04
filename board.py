import numpy as np
from card import BuildingCard, BuilderCard, CreatureCard
import constants as c
import pygame as pg
import draw


class Grid:
	def __init__(self, dimensions, origin, cell_size):
		self.dimensions = np.array(dimensions)
		self.origin = origin
		self.cell_size = cell_size
		self.update_drawable()
		self._generate_surface()

	@property
	def rect(self):
		return pg.Rect(self.origin, [(self.dimensions[x_n] * self.cell_size[x_n]) for x_n in range(2)])

	def get_cell_rect(self, cell):
		pos = self.get_cell_pos(cell=cell)
		return pg.Rect(pos, (self.cell_size[0]+1, self.cell_size[1]+1))

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
	# def get_cell_owner(self, cell):
	# 	if cell[1] >= 0 and cell[1] <= 2:
	# 		return 1
	# 	if cell [1] >= 3 and cell[1] <= 5:
	# 		return 0

	# distances format: {'up': 1, 'right': 2, ...}
	def get_cells_in_directions(self, start_cell, distances):
		cells = []
		if self.check_cell_valid(cell=start_cell) == False: return cells

		for direction, distance in distances.items():
			distance = max(0, distance)
			if direction == 'forward':
				dxs = [0] # difference between 		start_x and target_x
				dys = range(distance+1) # .. 	..	start_y and target_y
			elif direction == 'right':
				dxs = range(distance+1)
				dys = [0]
			elif direction == 'backward':
				dxs = [0]
				dys = range(0, -(distance+1), -1)
			elif direction == 'left':
				dxs = range(0, -(distance+1), -1)
				dys = [0]
			else:
				print('get_cells_in_directions received invalid direction key')
				return []

			start_x = start_cell[0]
			start_y = start_cell[1]
			cells += [(start_x+dx, start_y+dy) for dx in dxs for dy in dys if self.check_cell_valid(cell=(start_x+dx,start_y+dy)) == True]

		return list(set(cells)) # Remove duplicate cells

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
	def get_cell_pos(self, cell, align=('left','up')):
		pos = [self.rect[i] + cell[i]*self.cell_size[i] for i in range(2)]
		
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

	# def get_lane_at_pos(self, pos):
	# 	hit = False
	# 	cell_x = (pos[0] - self.rect.x) // self.cell_size[0]
	# 	cell_y = (pos[1] - self.rect.y) // self.cell_size[1]

	# 	if cell_x >= 0 and cell_x < self.dimensions[0] and cell_y >= 0 and cell_y < self.dimensions[1]:
	# 		hit = True

	# 	if hit == True:
	# 		return (cell_x, cell_y)
	# 	else:
	# 		return None

	def get_cell_at_pos(self, pos):
		hit = False
		x, y = pos

		grid_x = (x - self.rect.x) // self.cell_size[0]
		grid_y = (y - self.rect.y) // self.cell_size[1]

		if grid_x >=0 and grid_x < self.dimensions[0] and grid_y >= 0 and grid_y < self.dimensions[1]:
			hit = True

		if hit == True:
			return (grid_x, grid_y)
		else:
			return None

	def _generate_surface(self):
		self.surface = pg.Surface((self.rect.size[0]+1, self.rect.size[1]+1))
		pg.draw.rect(self.surface, c.dark_green, ((0,0), self.rect.size))
		# pg.draw.rect(self.surface, c.very_dark_blue, ((0,0), (self.rect.width, self.rect.height//2)))
		# pg.draw.rect(self.surface, c.very_dark_red, ((0,self.rect.height//2),(self.rect.width,self.rect.height//2)))

		grid_color = c.white

		for x in range(self.dimensions[0]+1):
			x_start = x*self.cell_size[0]
			pg.draw.line(self.surface, grid_color, (x_start, 0), (x_start, self.cell_size[1]*self.dimensions[1]))
		for y in range(self.dimensions[1]+1):
			y_start = y*self.cell_size[1]
			pg.draw.line(self.surface, grid_color, (0, y_start), (self.cell_size[0]*self.dimensions[0], y_start))		

	def draw(self, screen, color=c.white, player_perspective=0):
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

	def draw_surface_in_cell(self, source, cell, align=('left','up'), stretch=False, offset=(0,0)):
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

		cell_pos = self.get_cell_pos(cell, align)
		draw.draw_surface_aligned(target=screen, source=surface, pos=cell_pos, align=align, offset=offset)


class Board:
	def __init__(self, field, size):
		self.field = field

		self.size = size
		self.building_cards = np.full(size, None, np.dtype(BuildingCard))
		self.unit_cards = np.full(size, None, np.dtype(CreatureCard))
		grid_origin = (c.screen_size[0]//2-int((size[0]*c.node_size[0])//2), c.screen_size[1]//2-int((size[1]*c.node_size[1])//2))
		self.grid = Grid(dimensions=size, origin=grid_origin, cell_size=c.node_size)

		player_count = 2
		self._fow_visible_cells = [[] for player in range(player_count)]
		self._generate_default_fow_surfaces()


		self.selected_cell = None
		player_count = 2
		self.queued_cards = [[None]*c.grid_count[0] for i in range(player_count)]

	def __iter__(self):
		return iter([card_group[x,y] for x in range(self.size[0]) for y in range(self.size[1]) for card_group in (self.building_cards, self.unit_cards)])

	def is_cell_revealed(self, cell, player):
		if cell in self._fow_visible_cells[player]:
			return True
		else:
			return False

	# Returns the rank index for the nth rank, relative to player.
	# n=0 -> the rank closest to the player
	# n=1 -> second-closest rank
	# ...
	def get_nth_rank(self, n, player):
		if player == 0:
			return util.clamp(self.size[1]-1-n, 0, self.size[1]-1)
		elif player == 1:
			return util.clamp(n, 0, self.size[1]-1)
		else:
			print("Invalid player given to get_nth_rank()")
			return 0

	# Returns rank closest to player
	def get_front_rank(self, player):
		return self.get_nth_rank(n=0, player=player)

	def check_cell_valid(self, cell):
		return self.grid.check_cell_valid(cell=cell)

	def reveal_cells(self, cells, player):
		for cell in cells:
			if cell not in self._fow_visible_cells[player]:
				self._fow_visible_cells[player].append(cell)

		transparent_cell = pg.Surface((self.grid.cell_size[0]+1, self.grid.cell_size[1]+1))
		transparent_cell.fill(c.pink)
		for cell in cells:
			if player == 1:
				# Adjusted for player perspective
				relative_cell = (cell[0], self.size[1] - 1 - cell[1])
			else:
				relative_cell = cell

			cell_pos = self.grid.get_cell_pos(cell=relative_cell, align=('center','center'))
			cell_pos = [cell_pos[0] - self.grid.origin[0], cell_pos[1] - self.grid.origin[1]]
			draw.draw_surface_aligned(target=self.fow_surfaces[player], source=transparent_cell, pos=cell_pos, align=('center','center'))

	def refresh_fow(self):
		self._generate_default_fow_surfaces()
		for player in range(2):
			for card in self:
				if card != None and card.owner == player:
					self.reveal_cells(cells=card.visible_cells(), player=player)


	def _generate_default_fow_surfaces(self):
		self.fow_surfaces = [pg.Surface((self.grid.rect.width+1, self.grid.rect.height+1)) for player in range(2)]
		for surface in self.fow_surfaces:
			surface.fill(c.black)
			surface.set_alpha(200)
			# We'll draw c.pink squares on top of visible squares to remove the FOW
			surface.set_colorkey(c.pink)

		# The cells that are visible by default (closest 2 ranks to your side, as of writing this comment)
		p0_fow_default_visible_cells = [(x,y) for x in range(0,self.size[0]+1) for y in range(self.size[1]-2,self.size[1])]
		p1_fow_default_visible_cells = [(x,y) for x in range(0,self.size[0]+1) for y in range(0,2)]

		# for card in self:
		# 	if card != None and card.owner == player_perspective:
		# 		fow_visible_cells += card.visible_cells()

		self._fow_visible_cells[0] = p0_fow_default_visible_cells
		self._fow_visible_cells[1] = p1_fow_default_visible_cells

		transparent_cell = pg.Surface((self.grid.cell_size[0]+1, self.grid.cell_size[1]+1))
		transparent_cell.fill(c.pink)
		for player in range(2):
			for cell in self._fow_visible_cells[player]:
				if player == 1:
					# Adjusted for player perspective
					relative_cell = (cell[0], self.size[1] - 1 - cell[1])
				else:
					relative_cell = cell

				cell_pos = self.grid.get_cell_pos(cell=relative_cell, align=('center','center'))
				cell_pos = [cell_pos[0] - self.grid.origin[0], cell_pos[1] - self.grid.origin[1]]
				draw.draw_surface_aligned(target=self.fow_surfaces[player], source=transparent_cell, pos=cell_pos, align=('center','center'))

	@property
	def lane_count(self):
		return self.size[0]

	@property
	def rank_count(self):
		return self.size[1]

	def queue_building(self, cell, building_card, owner):
		self.place_card(cell=cell, card=building_card, owner=owner, sync=False)

	def place_card(self, cell, card, owner, sync):
		if self.check_cell_valid(cell) == True and card != None:
			card.place(board=self, cell=cell, owner=owner)

			if sync == True:
				send_string = 'card placed;' + card.name + ';' + str(cell[0]) + ';' + str(cell[1]) + ';' + str(owner) + ';[END]'
				self.field.game.queue_network_data(send_string.encode('utf-8'))

			self.refresh_fow()
			return True # Successfully fulfilled requirements for placing the card and placed it.
		else:
			print("Tried to place card in invalid cell or tried to place None card")
			return False

	def delete_unit_from_board(self, cell, sync):
		if self.unit_cards[cell] != None:
			self.unit_cards[cell] = None
			self.refresh_fow()

			if sync == True:
				send_string = 'unit deleted;' + str(cell[0]) + ';' + str(cell[1]) + ';[END]'
				self.field.game.queue_network_data(send_string.encode('utf-8'))

	def delete_building_from_board(self, cell, sync):
		if self.building_cards[cell] != None:
			self.building_cards[cell] = None
			self.refresh_fow()

			if sync == True:
				send_string = 'building deleted;' + str(cell[0]) + ';' + str(cell[1]) + ';[END]'
				self.field.game.queue_network_data(send_string.encode('utf-8'))

	def move_unit(self, start_cell, target_cell, sync):
		if self.check_cell_valid(start_cell) == False or self.check_cell_valid(target_cell) == False: return

		if self.unit_cards[target_cell] == None:
			card = self.unit_cards[start_cell]
			self.place_card(cell=target_cell, card=card, owner=card.owner, sync=sync)
			self.delete_unit_from_board(start_cell, sync=sync)

		if sync == True:
			send_string = 'unit moved;' + 	str(start_cell[0]) + ';' + str(start_cell[1]) + ';' + str(target_cell[0]) + ';' + str(target_cell[1]) + ';[END]'
			self.field.game.queue_network_data(send_string.encode('utf-8'))

	def move_building(self, start_cell, target_cell, sync):
		if self.building_cards[target_cell] == None:

			card = building_cards[start_cell]
			self.place_card(cell=target_cell, card=card, owner=card.owner, sync=sync)
			self.delete_building_from_board(start_cell, sync=sync)

		if sync == True:
			send_string = 'building moved;' + 	str(start_cell[0]) + ';' + str(start_cell[1]) + ';' + str(target_cell[0]) + ';' + str(target_cell[1]) + ';[END]'
			self.field.game.queue_network_data(send_string.encode('utf-8'))		



	def return_card_to_hand(self, cell):
		pass
		# if self.cards[cell] != None:
		# 	self.cards[cell].owner = None
		# 	card_name = self.cards[cell].name
		# 	self.field.active_hand.add_card(name=card_name)
		# 	self.cards[cell] = None
		# 	self._refresh_passives()

		# 	return card_name

	def fight_cards(self, attacker_cell, defender_cell, sync):
		try:
			attacker = self.unit_cards[attacker_cell]
			defender = self.unit_cards[defender_cell]

			defender.change_health(-attacker.power)
			attacker.change_health(-defender.power)

			if defender.health <= 0:
				self.delete_unit_from_board(cell=defender_cell, sync=sync)
			if attacker.health <= 0:
				self.delete_unit_from_board(cell=attacker_cell, sync=sync)

			if sync == True:
				send_string = 'cards fought;' + str(attacker_cell[0]) + ';' + str(attacker_cell[1]) + ';' + str(defender_cell[0]) + ';' + str(defender_cell[1]) + ';[END]'
				self.field.game.queue_network_data(send_string.encode('utf-8'))
		except IndexError:
			print('Tried to fight cards in invalid cells')
		except AttributeError:
			print('Tried to fight None card')


	# def get_frontmost_occupied_cell(self, player, lane):
	# 	ranks = range(self.)
	# 	if player == 0:
	# 		ranks = range(3,6,1) #3,4,5
	# 	elif player == 1:
	# 		ranks = range(2,-1,-1) #2,1,0
	# 	else:
	# 		print("get_frontmost_occupied_cell got invalid player")
	# 		return {'error:': True}

	# 	for rank in ranks:
	# 		if self.cards[lane, rank] != None:
	# 			return {'error': False,
	# 					'cell': (lane, rank)}

	# 	return {'error': False,
	# 			'cell': None} # There are no cards in the lane



	def _refresh_passives(self):
		pass 
		# dirty = False

		# self._clear_buffs()
		# self.do_passive()

		# if dirty == True:
		# 	self._refresh_passives() # Iterative refreshes when state has changed
		# 	# This is necessary because of the complex interactions between cards

	def _clear_buffs(self):
		pass
		# for _, card in np.ndenumerate(self.cards):
		# 	if card != None:
		# 		card.clear_buffs()


	def buff_creatures_in_range(self, power, max_health, cell, distance=1):
		pass
		# if cell:
		# 	cell_coords = self.grid.get_cells_by_distance(start_cell=cell, distance=distance)
		# 	for cell_coord in cell_coords:
		# 		if isinstance(self.cards[cell_coord], CreatureCard):
		# 			self.cards[cell_coord].apply_buff(power=1,max_health=1)

	def draw(self, screen, player_perspective=0):
		self.grid.draw(screen=screen, color=c.grey, player_perspective=player_perspective)

		screen.blit(self.fow_surfaces[player_perspective], self.grid.origin)
		if self.selected_cell != None:
			# Draw the c.green highlight around border of selected cell
			pg.draw.rect(screen, c.green, self.grid.get_cell_rect(self.selected_cell), 1)

		# Draw the cards in the board
		# For each type of card (unit, building, ...) loop through all cells and draw the card
		# (Draw unit cards on top of building cards)
		for card in self:
			if card != None:
				if card.cell in self._fow_visible_cells[player_perspective]:
					cell_x, cell_y = card.cell
					if player_perspective == 1:
						cell_y = self.size[1] - 1 - cell_y
					card_pos = self.grid.get_cell_pos((cell_x,cell_y), align=('center','top'))
					card_pos[0] -= c.board_card_size[0]//2
					card.draw(card_pos, 'board')