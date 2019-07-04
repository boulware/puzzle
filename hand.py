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
				self.cards.append(self.field.game.card_pool.get_card_by_name(name))
		else:
			d.print_callstack()
			print("Tried to add non-existent card to hand.")

	def pop_card(self, index):
		return self.cards.pop(index)

	def clear_hand(self):
		self.cards = []
