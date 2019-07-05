class GameObject:
	def __getitem__(self, attr_name):
		return getattr(self, attr_name)

	def __setitem__(self, attr_name, value):
		setattr(self, attr_name, value)