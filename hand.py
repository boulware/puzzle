import debug as d

class Hand:
	def __init__(self, field):
		self.cards = []
		self.field = field

	def __iter__(self):
		return iter(self.cards)

	def __getitem__(self, key):
		try:
			return self.cards[key]
		except IndexError as error:
			print(error)

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

	def add_random_cards(self, count=1):
		for _ in range(count):
			self.cards.append(self.field.game.card_pool.get_random_card())

class Deck:
	def __init__(self, field):
		self.field = field
		self.cards = []

	def add_random_cards(self, count):
		for i in range(count):
			self.cards.append(self.field.game.card_pool.get_random_card())
