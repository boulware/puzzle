import inspect
import colorama
from colorama import Fore, Back, Style
import traceback
import functools
import copy
import pygame as pg
import constants as c

from game_object import GameObject

colorama.init(autoreset=True)

print_callstack = traceback.print_stack
active = False
active_print = False
debugger = None

def info(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		if active_print is True:
			stack = inspect.stack()
			args_repr = [repr(a) for a in args]
			kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
			signature = ", ".join(args_repr + kwargs_repr)

			print(f"{inspect.getfile(func)}:{stack[2].lineno}->" + Fore.RED + f"{func.__name__}" + Fore.YELLOW + f"({signature})")
		return func(*args, **kwargs)

	return wrapper

class DebugUI(GameObject):
	def __init__(self, game, active=False):
		self.game = game
		self.active = active
		self.font = pg.font.Font('Montserrat-Regular.ttf', 14)
		self.bg_alpha = 0
		self.displayed_strings = []
		from UI import TreeView
		self.test_float = 1.0 # To test operations on floats in debug interface

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
		if self.game.state.__class__.__name__ == 'Field':
			board = self.game.state.board
			if board.selected_cell is not None:
				selected_unit = board.unit_cards[board.selected_cell]
				selected_building = board.building_cards[board.selected_cell]
				if selected_unit is not None:
					pass#self.write(f"unit: {selected_unit.name}")
				if selected_building is not None:
					pass#self.write(f"unit: {selected_building.name}")
			if self.game.state.drag_card is not None:
				pass#self.write(f"{self.game.state.drag_card.name}")

			for card in self.game.state.active_hand:
				pass#self.write(f"{card.name}; board={card.board}")

	def draw(self):
		if self.active == False: return
		screen = self.game.screen

		if self.bg_alpha > 0:
			bg = pg.Surface(screen.get_size())
			bg.set_alpha(self.bg_alpha)
			pg.draw.rect(bg, c.black, ((0,0),(screen.get_size())))
			screen.blit(bg, (0,0))
		pg.draw.rect(screen, c.blue_green, ((0,0),(200,100)))

		current_y = 0
		for string in self.displayed_strings:
			string_surface = self.font.render(string, True, self.text_color)
			screen.blit(string_surface, (0, current_y))
			current_y += self.font.get_linesize()

		self.test_treeview.parent_node = self.game
		self.test_treeview.draw(target=screen)

	def any_key_pressed(self, key, mod, unicode_key):
		self.test_treeview.any_key_pressed(key=key, mod=mod, unicode_key=unicode_key)
		if key == pg.K_d and mod == pg.KMOD_LCTRL:
			self.active = not self.active
		elif key == pg.K_p and mod == pg.KMOD_LCTRL:
			global active_print
			active_print = not active_print
		elif key == pg.K_t:
			for depth_pair in self.test_treeview.current_list:
				string = depth_pair[0]
				depth = depth_pair[1]
				print(Fore.RED + '>>'*depth + Fore.WHITE + ' ' + str(string))