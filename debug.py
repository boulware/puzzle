import inspect
import colorama
from colorama import Fore, Back, Style
import traceback
import functools
import copy
import pygame as pg
import constants as c
import time

colorama.init(autoreset=True)

print_callstack = traceback.print_stack
active = False
active_print = True
debugger = None

def info(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		if active_print is True:
			stack = inspect.stack()
			args_repr = [repr(a) for a in args]
			kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
			signature = ", ".join(args_repr + kwargs_repr)

			frame = inspect.stack()[1]
			module = inspect.getmodule(frame[0])
			filename = module.__file__
			print(f"{filename}:{stack[1].lineno}->" + Fore.RED + f"{func.__name__}" + Fore.YELLOW + f"({signature})")
		return func(*args, **kwargs)

	return wrapper

def time_me(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		start_time = time.clock()
		return_value = func(*args, **kwargs)
		end_time = time.clock()
		dt = end_time - start_time
		print(f"{func.__name__}: {dt*pow(10,3):.3f} ms")
		return return_value

	return wrapper

class DebugUI:
	def __init__(self, game, active=False):
		self.game = game
		self.active = active
		self.font = pg.font.Font('Montserrat-Regular.ttf', 14)
		self.bg_alpha = 0
		self.displayed_strings = []
		from UI import TreeView

		self.test_treeview = TreeView(pos=(0,0), font_size=14, parent_node_object=self)
		self.test_treeview.root.load_children() # Required for the treeview itself to show up in the debug list from the beginning

		self._hook_all()

	@property
	def active(self):
		return self._active

	@active.setter
	def active(self, value):
		self._active = value
		global active
		active = value

	def _hook_all(self):
		"""Hook into self.game and decorate its methods to enable debugging"""
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

	def print(self, values):
		for name, value in values.items():
			print(Fore.RED + f"{name}" + Fore.WHITE + f"=" + Fore.YELLOW + f"{repr(value)}")
			try:
				for key,e in vars(value).items():
					print('\t' + Fore.CYAN + f"{key}" + Fore.WHITE + ' = ' + Fore.GREEN + f"{repr(e)}")
			except TypeError:
				pass

		print(Fore.BLUE + "***********")

	def write(self, string):
		self.displayed_strings.append(string)

	def update(self, dt, mouse_pos):
		self.displayed_strings = []

	def draw(self):
		if self.active == False: return
		screen = self.game.screen

		if self.bg_alpha > 0:
			bg = pg.Surface(screen.get_size())
			bg.set_alpha(self.bg_alpha)
			pg.draw.rect(bg, c.black, ((0,0),(screen.get_size())))
			screen.blit(bg, (0,0))

		current_y = 0
		for string in self.displayed_strings:
			string_surface = self.font.render(string, True, self.text_color)
			screen.blit(string_surface, (0, current_y))
			current_y += self.font.get_linesize()

		self.test_treeview.parent_node = self.game
		self.test_treeview.draw(target=screen)

	def any_key_pressed(self, key, mod, unicode_key):
		if key == pg.K_d and mod == pg.KMOD_LCTRL:
			self.active = not self.active

		# Only allow the ctrl+d to enable or disable the debug interface if it's "deactivated"
		if self.active is True:
			self.test_treeview.any_key_pressed(key=key, mod=mod, unicode_key=unicode_key)
			if key == pg.K_PLUS and mod == pg.KMOD_LCTRL:
				global active_print
				active_print = not active_print
			elif key == pg.K_t and mod == pg.KMOD_LCTRL:
				if self.bg_alpha != 255:
					self.bg_alpha = 255
				else:
					self.bg_alpha = 0