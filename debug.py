import inspect
import colorama
from colorama import Fore, Back, Style
import traceback
import functools
import copy

import pygame as pg
import constants as c
import game_state

# import main

colorama.init(autoreset=True)

print_callstack = traceback.print_stack

def info(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		stack = inspect.stack()
		args_repr = [repr(a) for a in args]
		kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
		signature = ", ".join(args_repr + kwargs_repr)

		print(f"{inspect.getfile(func)}:{stack[2].lineno}->" + Fore.RED + f"{func.__name__}" + Fore.YELLOW + f"({signature})")
		return func(*args, **kwargs)

	return wrapper

class DebugUI:
	def __init__(self, game, active=False):
		self.game = game
		self.active = active
		self.font = pg.font.Font('Montserrat-Regular.ttf', 16)

		self._hook_all()

	# Hook into self.game and decorate its methods to enable debugging
	def _hook_all(self):
		self.game.draw = self._draw_hook(hooked_func=self.game.draw)
		self.game.any_key_pressed = self._any_key_pressed_hook(hooked_func=self.game.any_key_pressed)
		self.game.update = self._update_hook(hooked_func=self.game.update)

	def _draw_hook(self, hooked_func):
		@functools.wraps(hooked_func)
		def wrapper(*args, **kwargs):
			hooked_func()
			self.draw()
		return wrapper

	def _any_key_pressed_hook(self, hooked_func):
		@functools.wraps(hooked_func)
		def wrapper(*args, **kwargs):
			self.any_key_pressed(*args, **kwargs)
			hooked_func(*args, **kwargs)
		return wrapper

	def _update_hook(self, hooked_func):
		@functools.wraps(hooked_func)
		def wrapper(*args, **kwargs):
			hooked_func(*args, **kwargs)
			self.update(*args, **kwargs)
		return wrapper		

	def update(self, dt, mouse_pos):
		if isinstance(self.game.state, game_state.Field):
			print('field')

	def draw(self):
		if self.active == False: return
		screen = self.game.screen

		pg.draw.rect(screen, c.blue_green, ((0,0),(50,100)))

	def any_key_pressed(self, key, mod, unicode_key):
		if key == pg.K_d and mod == pg.KMOD_LCTRL:
			self.active = not self.active