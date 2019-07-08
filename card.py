import debug as d
from card_actions import *
from card_components import *
import pygame as pg
import constants as c
from font import fonts
import draw
import copy

class Card:
	def __init__(self, name, cost, visibility, default_action=None):
		self.name = name
		self.cost = cost
		self.board = None
		self._cell = None
		self.previous_cell = None # Used for sliding animatino between cells when moving
		self._owner = None

		self.active_actions = {}
		self.default_action = default_action
		self.current_action = default_action

		self.dirty = True

		self._hand_surface = None
		self._board_surface = None
		self._action_panel_surface = None

		self._health_component = None
		self._attack_component = None

		self._power = 0

		self.activated = True

		# Number of values in visibility
		# 1 value -> the unit sees that value in all (cardinal?) directions
		# 4 values-> the unit sees in each cardinal direction those 4 values.
		#			First value is in their face direction, second is clockwise from there, etc.
		# Anything else is invalid (for now; I might add 8-directions later)
		vis_dirs = len(visibility)
		if vis_dirs != 1 and vis_dirs != 4:
			print('invalid visibility given to Card(). Using default vis=[1]')
			self.visibility = [1]
		else:
			self.visibility = visibility

		self.queue_lane = None

	@property
	def cell(self):
		return self._cell

	@cell.setter
	def cell(self, value):
		self.previous_cell = self._cell
		self._cell = value

	def remove_from_board(self):
		self.board = None
		self.cell = None
		self.activated = False

	def queue(self, board, cell, owner):
		"""Places the card in the appropriate queue_lane for the given board, cell, and owner"""
		lane = cell[0]
		active_queue = board.queued_cards[owner] # Queue of the active player
		previous_card = active_queue[lane] # This is the card that was previously in the queue ane, to be removed

		# If there's a card already in the queue, unqueue it
		if previous_card != None:
			previous_card.unqueue()

		active_queue[lane] = self
		self.board = board
		self.owner = owner
		self.queue_lane = lane

	def unqueue(self):
		if self.queue_lane is not None:
			self.board.queued_cards[self.owner][self.queue_lane] = None
			self.board.field.active_hand.add_card(name=self.name)

	def place_queued(self):
		if self.queue_lane == None: return

		active_queue = self.board.queued_cards[self.owner]
		target_rank = self.board.get_front_rank(player=self.owner)
		target_lane = self.queue_lane
		target_cell = (target_lane, target_rank)

		# If the target cell is empty, place this card there; otherwise, do nothing
		if self.board.get_unit_in_cell(cell=target_cell) is None:
			self.board.set_unit_in_cell(cell=target_cell, card=self)
			self.cell = target_cell
			self.queue_lane = None
			self.activated = True

	def place(self, board, cell, owner):
		self.cell = cell
		self.board = board
		self.owner = owner

		self.activated = True

	def act(self):
		if self.current_action == None: return
		if self.activated == False: return

		if self.current_action in self.active_actions:
			self.active_actions[self.current_action].do()
		else:
			print("Tried to do invalid action: ", self.current_action)

	@property
	def target_cell(self):
		return self._target_cell

	@target_cell.setter
	def target_cell(self, value):
		self._target_cell = value
		for k, action in self.active_actions.items():
			action.target_cell = value

	def visible_cells(self):
		if self.cell == None: return []

		# Alter the directions to make visibility work in the correct direction
		# TODO: Make this less janky
		if self.owner == 0:
			directions = ('backward', 'left', 'forward', 'right')
		else:
			directions = ('forward', 'right', 'backward', 'left')

		if len(self.visibility) == 1:
			distance = self.visibility[0]
			distance_dict = {direction:distance for direction in directions}
			cells = self.board.grid.get_cells_in_directions(start_cell=self.cell,
															distances=distance_dict)
		elif len(self.visibility) == 4:
			distances = self.visibility
			distance_dict = dict(zip(directions, distances))
			cells = self.board.grid.get_cells_in_directions(start_cell=self.cell,
															distances=distance_dict)
		else:
			cells = [self.cell]

		return cells

	@property
	def health(self):
		return self._health_component.health

	@health.setter
	def health(self, health):
		if self._health_component != None:
			self.dirty = True
			self._health_component.health = health

	def change_health(self, amount):
		if self._health_component != None:
			self.dirty = True
			self._health_component.change_health(amount)

	@property
	def max_health(self):
		if self._health_component != None:
			return self._health_component.max_health

	@max_health.setter
	def max_health(self, value):
		if self._health_component != None:
			self.dirty = True
			self._health_component.max_health = value

	@property
	def power(self):
		return self._power

	@power.setter
	def power(self, value):
		self.dirty = True
		self._power = value

	@property
	def cost(self):
		return self._cost

	@cost.setter
	def cost(self, value):
		self.dirty = True
		self._cost = value

	def move_to_hand(self, hand):
		if isinstance(self.sub_board, np.ndarray) == False:
			print('sub_board not set')
			d.print_callstack()
			return

		if self.board == True:
			# This should always be False, but check just in case
			if self != self.sub_board[self.cell]:
				print('something about board/cell state in card is wrong')
				return

			self.sub_board[self.cell] = None
			# These don't really matter atm because hand.add_card() pulls a copy from card pool, but futureproofing
			self.board = None
			self.cell = None

		hand.add_card(name=self.name)

	@property
	def hand_surface(self):
		if self._hand_surface == None or self.dirty is True:
			self.dirty = False
			self.generate_surfaces()

		return self._hand_surface

	@property
	def board_surface(self):
		if self._board_surface == None or self.dirty is True:
			self.dirty = False
			self.generate_surfaces()

		return self._board_surface

	def _generate_hand_surface(self):
		bg_color = c.dark_grey
		if self.owner == 0:
			bg_color = c.dark_red
		elif self.owner == 1:
			bg_color = c.dark_blue

		self._hand_surface = pg.Surface(c.hand_card_size)

		pg.draw.rect(self.hand_surface, bg_color, ((0,0), c.hand_card_size))
		pg.draw.rect(self.hand_surface, c.light_grey, ((0,0), c.hand_card_size), 1)
		title_surface = fonts.card_text_sm.render(self.name, True, c.white)
		self.hand_surface.blit(title_surface, (5,0))
		cost_surface = fonts.card_text_lg.render(str(self.cost), True, c.light_grey)
		draw.draw_surface_aligned(target=self.hand_surface, source=cost_surface, pos=self.hand_surface.get_rect().center, align=('center','center'))

		if self._health_component != None:
			# Draw health bar
			draw.draw_surface_aligned(	target=self.hand_surface,
										source=self._health_component.health_bar.surface,
										pos=c.hand_card_size,
										align=('right','down'),
										offset=(-1,-1))

	def _generate_board_surface(self):
		self._board_surface = pg.transform.smoothscale(self.hand_surface, c.board_card_size)

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
	def board(self):
		return self._board

	@board.setter
	def board(self, value):
		self._board = value

	@property
	def owner(self):
		return self._owner

	@owner.setter
	def owner(self, owner):
		self._owner = owner
		self.generate_surfaces()

	def clone(self):
		raise NotImplementedError()

	def apply_buff(self):
		pass

	def clear_buffs(self):
		self.buffs = []
		self.dirty = True

	def end_turn(self):
		for _, action in self.active_actions.items():
			action.end_turn()

	def draw(self, pos, location, hover=False):
		if location == 'hand':
			draw.screen.blit(self.hand_surface, pos)
			if hover:
				pg.draw.rect(draw.screen, c.gold, (pos, self.hand_surface.get_size()), 3)
		elif location == 'board' or location == 'board_hover':
			draw.screen.blit(self.board_surface, pos)
		elif location == 'queue':
			draw.screen.blit(self.board_surface, pos)

class BuildingCard(Card):
	def __init__(self, name, cost, max_health, visibility, health=None, default_action=None, active_actions={}, enter_fns={}):
		Card.__init__(self=self, name=name, cost=cost, visibility=visibility, default_action=default_action)

		self._health_component = HealthComponent(max_health=max_health, health=health)
		self.active_actions.update(active_actions)
		for key in self.active_actions:
			self.active_actions[key] = self.active_actions[key](card=self)

		self.enter_fns = enter_fns

	# def place(self, board, sub_board, cell, owner):
	# 	Card.place(self=self, board=board, sub_board=self.sub_board, cell=cell, owner=owner)

	def queue(self, board, cell, owner):
		lane = cell[0]
		active_queue = board.queued_cards[owner] # Queue of the active player
		previous_card = active_queue[lane] # Card in the lane queue already

		# If there's a card already in the queue, unqueue it
		if previous_card != None:
			previous_card.unqueue()

		drone = board.field.game.card_pool.get_card_by_name(name='Drone')
		drone.target_cell = cell
		drone.target_building = self
		drone.queue(board=board, cell=cell, owner=owner)

		self.cell = cell
		self.board = board
		self.owner = owner
		#self.sub_board[cell] = self
		self.activated = False

	# TODO: Add network sync; the problem is I've been relying on board.place_card, but this doesn't go through that method
	def complete(self):
		self.activated = True
		self.board.refresh_fow()

	def visible_cells(self):
		if self.activated == False: return [] # Don't show on the map if the building isn't built yet
		return Card.visible_cells(self)


	@property
	def activated(self):
		return self._activated

	@activated.setter
	def activated(self, value):
		self._activated = value
		self._generate_board_surface()

	def _generate_board_surface(self):
		Card._generate_board_surface(self)

		if self.activated == False:
			pending_text_surface = fonts.card_text_v_sm.render('Pending', True, c.green)
			pending_surface = pg.Surface((self._board_surface.get_width(), pending_text_surface.get_height()))
			pending_surface.blit(pending_text_surface, (pending_surface.get_rect().centerx - pending_text_surface.get_width()//2,0))

			pg.draw.rect(pending_surface, c.green, pending_surface.get_rect(), 1)

			# a surface that fades out the normal parts of the card while it is pending
			black_out_surface = pg.Surface(self._board_surface.get_size())
			black_out_surface.set_alpha(120)

			self._board_surface.blit(black_out_surface, (0,0))
			self._board_surface.blit(pending_surface, (0,self._board_surface.get_height()//2 - pending_surface.get_height()//2))

	def clone(self):
		return BuildingCard(name = self.name,
							cost = self.cost,
							max_health = self.max_health,
							health = self.health,
							visibility = self.visibility,
							default_action = self.default_action,
							enter_fns = copy.copy(self.enter_fns),
							active_actions = copy.copy(self.active_actions))

class CreatureCard(Card):
	def __init__(self, name, cost, base_power, max_health, visibility, health=None):
		Card.__init__(self=self, name=name, visibility=visibility, cost=cost)

		self.active_actions.update({'M': MoveAction(card=self), 'AM': AttackMoveAction(card=self)})
		self.current_action = 'AM'

		self.power = base_power
		self._health_component = HealthComponent(max_health=max_health, health=health)

	# Returns the cell directly in front of the card (assuming it's facing towards the enemy)
	def get_front_cell(self):
		if self.cell == None: return None

		if self.owner == 0:
			target_cell = (self.cell[0], self.cell[1]-1)
		else:
			target_cell = (self.cell[0], self.cell[1]+1)

		return target_cell

	def attack_move(self):
		target_cell = self.get_front_cell()
		# TODO: This only works if the unit can only move 'forward'
		if self.board.check_cell_valid(cell=target_cell) == False:
			# The unit is at the farthest cell (so it should deal damage to the enemy)
			self.board.delete_unit_from_board(cell=self.cell, sync=True)
			self.board.field.change_health(amount=-self.power, player=self.enemy, sync=True)

		# the card occupying the target_cell (if nothing does, None)
		target_card = self.board.unit_cards[target_cell]
		if target_card == None:
			# There's no unit in the way; just move to the cell
			self.board.move_unit(start_cell=self.cell, target_cell=target_cell, sync=True)
		else:
			# There's a unit in the way; attack if it's owned by the enemy
			if target_card.owner != self.owner:
				self.board.fight_cards(attacker_cell=self.cell, defender_cell=target_cell, sync=True)


	# @property
	# def board(self):
	# 	return self._board

	# @board.setter
	# def board(self, value):
	# 	self._board = value

	# @property
	# def sub_board(self):
	# 	if self.board == None: return
	# 	return self.board.unit_cards

	def _generate_hand_surface(self):
		Card._generate_hand_surface(self)

		# Draw power value
		power_text = fonts.card_text_sm.render(str(self.power), True, c.green)
		bottomleft = self.hand_surface.get_rect().bottomleft
		draw.draw_surface_aligned(	target=self.hand_surface,
								source=power_text,
								pos=bottomleft,
								align=('left','down'),
								offset=(6,-4))

		health_text = fonts.card_text_med.render(str(self.health), True, c.red)
		draw.draw_surface_aligned(	target=self.hand_surface,
								source=health_text,
								pos=c.hand_card_size,
								align=('right','down'),
								offset=(-20,1))


	def _generate_board_surface(self):
		Card._generate_board_surface(self)

	def generate_surfaces(self):
		self._generate_hand_surface()
		self._generate_board_surface()

	def apply_buff(self, power=0, max_health=0):
		pass
		# buff = (power,max_health)
		# if buff != (0,0):
		# 	self.dirty = True
		# 	self.buffs.append(buff)
		# 	self.health_bar.max_health = self.max_health

	def clear_buffs(self):
		pass
		# Card.clear_buffs(self)
		# self.health_bar.max_health = self.max_health
		# if board.check_if_card_is_front(self.cell) == True:
		# 	game.change_health(-self.power, self.enemy)

	def clone(self):
		return CreatureCard(name = self.name,
							cost = self.cost,
							base_power = self.power,
							max_health = self.max_health,
							health = self.health,
							visibility = self.visibility)

# TODO: Make building a component rather than a subclass
class BuilderCard(CreatureCard):
	def __init__(self, name, cost, base_power, max_health, visibility, health=None):
		CreatureCard.__init__(	self=self, name=name, cost=cost,
								base_power=base_power, max_health=max_health,
								visibility=visibility, health=health)

		self.active_actions.update({'MB': MoveBuildAction(card=self)})
		self.current_action = 'MB'

		self.base_power = base_power
		self._health_component = HealthComponent(max_health=max_health, health=health)

	def unqueue(self):
		if self.queue_lane is not None:
			self.board.queued_cards[self.owner][self.queue_lane] = None
			self.board.field.active_hand.add_card(name=self.target_building.name)

	@property
	def target_building(self):
		return self.active_actions['MB'].target_building

	@target_building.setter
	def target_building(self, value):
		self.active_actions['MB'].target_building = value

	@property
	def target_cell(self):
		return self.active_actions['MB'].target_cell

	@target_cell.setter
	def target_cell(self, value):
		self.active_actions['MB'].target_cell = value

	def move_to_hand(self, hand):
		if self.sub_board == None:
			print('sub_board not set')
			return

		if self.board == True:
			# This should always be False, but check just in case
			if self != self.sub_board[self.cell]:
				print('something about board/pos state in card is wrong')
				return

			self.sub_board[self.cell] = None
			# These don't really matter atm because hand.add_card() pulls a copy from card pool, but futureproofing
			self.board = None
			self.cell = None

			# Builder card should just be deleted, and not returned to hand

	def draw(self, pos, location, hover=False):
		# Draw target pending building
		if location == 'board' or location == 'queue':
			building_pos = self.board.grid.get_cell_pos(cell=self.target_cell, align=('left','up'))

			#if self.board.field.player_number == self.owner:
			self.target_building.draw(pos=building_pos, location='board')

		Card.draw(self=self, pos=pos, location=location, hover=hover)

	def clone(self):
		return BuilderCard(	name = self.name,
							cost = self.cost,
							base_power = self.base_power,
							max_health = self.max_health,
							visibility = self.visibility,
							health = self.health)
