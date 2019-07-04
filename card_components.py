import util
import pygame as pg
import constants as c
import numpy as np

class HealthBar:
	def __init__(self, max_health, size, health=None):
		self.max_health = max_health
		self.size = size

		if health == None:
			health = max_health
		self._health = health

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
		pg.draw.rect(self.surface, c.red, (0, self.size[1]-red_height, self.size[0], self.size[1]))

		pg.draw.line(self.surface, c.white, (0,0), (0,self.size[1])) # draw left edge
		pg.draw.line(self.surface, c.white, (self.size[0],0), (self.size[0], self.size[1])) # draw right edge

		# draw borders which delineate cells (max_health+1 because we're drawing borders, not the cells themselves)
		for y in np.linspace(0, self.size[1], self.max_health+1):
			pg.draw.line(self.surface, c.white, (0,y), (self.size[0],y))

	def _clamp_health(self):
		self.health = util.clamp(self.health, 0, self.max_health)

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
		print('old health=', self._health)
		self.dirty = True
		self._health = health
		print('new health=', self._health)
	
	def set_health(self, new_health):
		self.dirty = True

		self.health = new_health
		self._clamp_health()

class HealthComponent:
	def __init__(self, max_health, health=None):
		self.max_health = max_health
		if health == None:
			health = max_health

		self._health = health
		self.health_bar = HealthBar(max_health=max_health, size=(15,80), health=health)
		self._clamp_health()

	@property
	def health(self):
		return self._health

	@health.setter
	def health(self, value):
		self._health = value
		self._clamp_health()
		self.health_bar.set_health(self._health)

	def _clamp_health(self):
		self._health = util.clamp(self._health, 0, self.max_health)

	def change_health(self, amount):
		self.health += amount

	def draw(self, pos):
		screen.blit(self.health_bar.surface, pos)
