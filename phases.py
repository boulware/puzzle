import constants as c

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
			self.phase_texts.append(self.font.render(name, True, c.white))
			self.phase_active_texts.append(self.font.render(name, True, c.green))

	def draw(self, target, pos):
		line_spacing = self.font.get_linesize()
		for phase_ID, _ in enumerate(self.phase):
			if phase_ID == self.phase.ID:
				text_set = self.phase_active_texts # Draw from the active text set
			else:
				text_set = self.phase_texts # Draw from non-active text set

			target.blit(text_set[phase_ID], (pos[0],pos[1]+line_spacing*phase_ID))
