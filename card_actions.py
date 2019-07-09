import debug as d
import animation as anim
import constants as c

class Action:
	def __init__(self, card):
		self.card = card
		self.surface = None

	@property
	def target_cell(self):
		return self._target_cell

	@target_cell.setter
	def target_cell(self, value):
		self._target_cell = value

	@property
	def board(self):
		return self.card.board

	@property
	def owner(self):
		return self.card.owner

	def end_turn(self):
		pass

	def do(self):
		raise NotImplementedError()

class MoveAction(Action):
	def __init__(self, card):
		Action.__init__(self=self, card=card)

	@property
	def target_cell(self):
		if self.board == None: return

		return self.card.get_front_cell()

	@target_cell.setter
	def target_cell(self, value):
		pass

	@d.info
	def do(self):
		if self.target_cell == None: return

		if self.board.check_cell_valid(cell=self.target_cell) == False:
			pass # If we're trying to move into an invalid cell, do nothing

		target_card = self.board.get_unit_in_cell(cell=self.target_cell)
		if target_card is None:
			# There's no unit in the way; just move to the cell

			start_pos = self.board.grid.get_cell_pos(cell=self.card.cell)
			start_pos[0] += c.board_card_size[0]
			end_pos = self.board.grid.get_cell_pos(cell=self.target_cell)
			end_pos[0] += c.board_card_size[0]
			self.card.current_animation = anim.MoveAnimation(start_pos=start_pos, end_pos=end_pos, frame_duration=30)

			self.board.move_unit(start_cell=self.card.cell, target_cell=self.target_cell, sync=True)
		else:
			pass # If there's a unit in the way and we're just moving (not attack moving), do nothing

class AttackAction(Action):
	def __init__(self, card):
		Action.__init__(self=self, card=card)

	@property
	def target_cell(self):
		if self.board == None: return

		return self.card.get_front_cell()

	def do(self):
		if self.target_cell == None: return

		if self.board.check_cell_valid(cell=self.target_cell) == False:
			pass

		target_card = self.board.get_unit_in_cell(cell=self.target_cell)
		if target_card is not None:
			if target_card.owner != self.owner:

				start_pos = self.board.grid.get_cell_pos(cell=self.card.cell)
				start_pos[0] += c.board_card_size[0]
				target_pos = self.board.grid.get_cell_pos(cell=self.target_cell)
				target_pos[0] += c.board_card_size[0]
				self.card.current_animation = anim.AttackAnimation(start_pos=start_pos, target_pos=target_pos)
				target_card.current_animation = anim.AttackAnimation(start_pos=target_pos, target_pos=start_pos)
				self.board.fight_cards(attacker_cell=self.card.cell, defender_cell=target_card.cell, sync=True)


class AttackMoveAction(Action):
	def __init__(self, card):
		Action.__init__(self=self, card=card)

		self.move_action = MoveAction(card=card)
		self.attack_action = AttackAction(card=card)

	@property
	def target_cell(self):
		if self.board == None: return

		return self.card.get_front_cell()

	@target_cell.setter
	def target_cell(self, value):
		pass

	def do(self):
		# TODO: This check only works if the unit can only move 'forward'
		if self.board.check_cell_valid(cell=self.target_cell) is False:
			# The unit is at the farthest cell (so it should deal damage to the enemy)
			self.board.field.change_health(amount=-self.card.power, player=self.card.enemy, sync=True)
			self.board.delete_unit_from_board(cell=self.card.cell, sync=True)
		else:
			target_card = self.board.get_unit_in_cell(cell=self.target_cell)
			if target_card is None:
				# There's no unit in the way; just move to the cell
				self.move_action.do()
				#self.board.move_unit(start_cell=self.card.cell, target_cell=self.target_cell, sync=True)
			else:
				# There's a unit in the way; attack if it's owned by the enemy
				self.attack_action.do()
				# if target_card.owner != self.owner:
				# 	self.board.fight_cards(attacker_cell=self.card.cell, defender_cell=self.target_cell, sync=True)

class BuildAction(Action):
	def __init__(self, card, target_cell=None, target_building=None):
		Action.__init__(self=self, card=card)
		self.target_cell = target_cell
		self.target_building = target_building

	def do(self):
		if self.card.cell == self.target_cell:
			if self.board.building_cards[self.target_cell] is None:
				self.board.place_card(cell=self.target_cell, card=self.target_building, owner=self.owner, sync=True)
			self.board.delete_unit_from_board(cell=self.card.cell, sync=True) # Delete builder from board

class MoveBuildAction(Action):
	def __init__(self, card, target_cell=None, target_building=None):
		Action.__init__(self=self, card=card)

		self.move_action = MoveAction(card=card)
		self.build_action = BuildAction(card=card, target_cell=target_cell)
		self.target_cell = target_cell
		self.target_building = target_building

	@property
	def target_building(self):
		return self._target_building

	@target_building.setter
	def target_building(self, value):
		self._target_building = value
		self.move_action.target_building = value
		self.build_action.target_building = value

	@property
	def target_cell(self):
		return self._target_cell

	@target_cell.setter
	def target_cell(self, value):
		self._target_cell = value
		self.move_action.target_cell = value
		self.build_action.target_cell = value

	def do(self):
		if self.card.cell == None: return

		if self.board.building_cards[self.target_cell] is not None:
			print(f'deleting {self.card.name} from {self.card.cell}')
			self.board.delete_unit_from_board(cell=self.card.cell, sync=True)

		if self.card.cell == self.target_cell:
			self.build_action.do()
		else:
			self.move_action.do()

class ScanAction(Action):
	def __init__(self, card, target_cell=None):
		Action.__init__(self=self, card=card)
		self.target_cell = target_cell
		self.scans_per_turn = 1
		self.scans_this_turn = 0

	def end_turn(self):
		self.scans_this_turn = 0

	def do(self):
		if self.target_cell == None: return
		if self.scans_this_turn >= self.scans_per_turn: return
		if self.board.is_cell_revealed(cell=self.target_cell, player=self.owner) == True: return

		self.board.reveal_cells(cells=[self.target_cell], player=self.owner)
		self.scans_this_turn += 1

class MoveAttackBuildAction(Action):
	def __init__(self, card, target_cell=None, target_building=None):
		Action.__init__(self=self, card=card)

		self.attack_move_action = AttackMoveAction(card=card)
		self.build_action = BuildAction(card=card, target_cell=target_cell, target_building=target_building)
		self.target_cell = target_cell
		self.target_building = target_building

	@property
	def target_building(self):
		return self._target_building

	@target_building.setter
	def target_building(self, value):
		self._target_building = value
		self.build_action.target_building = value

	@property
	def target_cell(self):
		return self._target_cell

	@target_cell.setter
	def target_cell(self, value):
		self._target_cell = value
		self.attack_move_action.target_cell = value
		self.build_action.target_cell = value

	def do(self):
		if self.card.cell == None: return

		if self.board.building_cards[self.target_cell] is not None:
			print(f'deleting {self.card.name} from {self.card.cell}')
			self.board.delete_unit_from_board(cell=self.card.cell, sync=True)

		if self.card.cell == self.target_cell:
			self.build_action.do()
		else:
			self.attack_move_action.do()
