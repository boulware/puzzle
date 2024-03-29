import constants as c
import pygame as pg
import draw
import util
import numpy as np
import debug as d

import inspect

import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)


class Element:
	def __init__(self, parent_container=None):
		self.parent_container = parent_container
		self.events = []
	def get_event(self):
		if len(self.events) > 0:
			return self.events.pop(0)
	def any_key_pressed(self, key, mod, unicode_key):
		pass
	def left_mouse_pressed(self, mouse_pos):
		pass
	def left_mouse_released(self, mouse_pos):
		pass
	def update(self, dt, mouse_pos):
		pass

class Container:
	def __init__(self):
		self.elements = []
		self.group_parent = None
		self.focused_element = None

	def __iter__(self):
		return iter(self.elements)

	def add_element(self, element):
		self.elements.append(element)
		element.parent_container = self
		if self.focused_element == None:
			self.focused_element = element

	def focus_element(self, target):
		for e in self.elements:
			if e == target:
				self.focused_element = target
				if self.group_parent:
					self.group_parent.focus_element(target)
				return True
		return False
		# If the element isn't in this container, ignore the focus request

	# Returns True if there is no focused element upon return,
	# Returns False if there is still a focused element upon return
	def unfocus_element(self, target=None):
		if target == None:
			self.focused_element = None
		else:
			if target == self.focused_element:
				self.focused_element = None

		for e in self.elements:
			if isinstance(e, Container):
				e.unfocus_element(target)

	def any_key_pressed(self, key, mod, unicode_key):
		if self.focused_element:
			self.focused_element.any_key_pressed(key, mod, unicode_key)

	def left_mouse_pressed(self, mouse_pos):
		for e in self.elements:
			e.left_mouse_pressed(mouse_pos)

	def left_mouse_released(self, mouse_pos):
		if self.focused_element:
			self.focused_element.left_mouse_released(mouse_pos)

	def update(self, dt, mouse_pos):
		for element in self.elements:
			element.update(dt, mouse_pos)

	def draw(self, screen):
		for e in self.elements:
			e.draw(screen=screen)
			if e == self.focused_element:
				pass#pg.draw.circle(screen, pink, e.pos, 10)

# Represents a group of individual Containers that are displayed
# on the screen at once and interact smoothly between each other
class Group:
	def __init__(self, containers, screen):
		self.containers = containers
		self.focused_container = None
		self.screen = screen
		for container in self.containers:
			container.unfocus_element()
			container.group_parent = self

	def __iter__(self):
		elements = []
		for container in self.containers:
			for e in container:
				elements.append(e)

		return iter(elements)

	def focus_element(self, target):
		for container in self.containers:
			for e in container:
				if e == target:
					container.focused_element = target
					self.focused_container = container
		for container in self.containers:
			if container != self.focused_container:
				container.unfocus_element()

	def unfocus_element(self, target=None):
		for container in self.containers:
			container.unfocus_element(target)
		self.focused_container = None


	def any_key_pressed(self, key, mod, unicode_key):
		for container in self.containers:
			container.any_key_pressed(key, mod, unicode_key)

	def left_mouse_pressed(self, mouse_pos):
		for container in self.containers:
			container.left_mouse_pressed(mouse_pos)

	def left_mouse_released(self, mouse_pos):
		for container in self.containers:
			container.left_mouse_released(mouse_pos)

	def update(self, dt, mouse_pos):
		for container in self.containers:
			container.update(dt, mouse_pos)

	def draw(self):
		for container in self.containers:
			container.draw(self.screen)

class Label(Element):
	def __init__(self, pos, font, align=('left','top'), text_color=c.white, text='', parent_container=None):
		Element.__init__(self, parent_container)
		self.pos = pos
		self.font = font
		self.align = align
		self.text_color = text_color
		self.text = text

		self._generate_surface()

	@property
	def text(self):
		return self._text

	@text.setter
	def text(self, new_text):
		self._text = new_text
		self._generate_surface()


	def _generate_surface(self):
		self.surface = self.font.render(self.text, True, self.text_color)

	def draw(self, screen):
		draw.draw_surface_aligned(target=screen, source=self.surface, pos=self.pos, align=self.align)

class Node:
	def __init__(self, name, value, parent=None, expanded=False):
		self.name = name
		self.value = value
		self.parent = parent
		self.expanded = expanded

		self.children = []

		if parent is not None:
			parent.add_child(self)

	def set_value(self, value):
		self.value = value

		try:
			if isinstance(self.parent.value, (list,dict,np.ndarray)):
				self.parent.value[self.name] = value
			else:
				setattr(self.parent.value, self.name, value)
		except AttributeError as error:
			# Attribute has no setter, so do nothing
			pass
		except TypeError as error:
			# try:
				#print(f'lvl 2 (name={self.parent.name}): {e}')
				# The parent type doesn't support __setitem__ (as indexing), so we assume it's a tuple?
				# I think I'll ducktype all my classes for __getitem__ and __setitem__ to call getattr()/setattr()

				# If the parent is a tuple, we have to go one level above to search for a mutable type (list, dict, obj):
				# Then after we look one level up, if it's still not mutable, we go up a level again, u.s.w

				node = self
				parent = self.parent
				temp_list = list(parent.value)
				temp_list[node.name] = value

				mutable_found = False
				while mutable_found is False:
					node = parent
					parent = node.parent
					try:
						if isinstance(parent.value, (list,dict,np.ndarray,tuple)):
							parent.value[node.name] = tuple(temp_list)
						else:
							setattr(parent.value, node.name, tuple(temp_list))

						mutable_found = True
					except TypeError as error:
						new_temp_list = list(parent.value)
						if isinstance(parent.value, (list,dict,np.ndarray,tuple)):
							new_temp_list[parent.name] = tuple(temp_list)
						else:
							setattr(parent.value, parent.name, tuple(temp_list))
						temp_list = new_temp_list
			# except (TypeError, AttributeError) as error:
			# 	d.print_callstack()
			# 	print(Fore.YELLOW + f"Unable to set attribute \'{node.name}\' on {parent.value}")
			# 	print(Fore.RED + f"{type(parent.value).__name__} may not be ducktyped for __getattr__() or __setattr__()")
			# 	print(Fore.MAGENTA + f"error: {error}")

	def add_child(self, child):
		self.children.append(child)

	def get_indexable_attributes(self):
		# Returns list of attributes accessible by [] ('__getitem__')

		attribute_names = []

		if isinstance(self.value, (tuple,list)):
			for index, _ in enumerate(self.value):
				attribute_names.append(index)
		elif isinstance(self.value, np.ndarray):
			for index, _ in np.ndenumerate(self.value):
				attribute_names.append(index)
		elif isinstance(self.value, dict):
			for key, value in self.value.items():
				attribute_names.append(key)
		else:
			try:
				members = inspect.getmembers(self.value, lambda a:not(inspect.isroutine(a)))
				for member_name, _ in [a for a in members if not(a[0].startswith('__') and a[0].endswith('__'))]:
					attribute_names.append(member_name)
			except TypeError as e:
				attribute_names = []

		return attribute_names

	def load_children(self):
		self.children = []

		attribute_names = self.get_indexable_attributes()

		if len(attribute_names) == 0:
			self.expanded = False

		if isinstance(self.value, (list,dict,np.ndarray,tuple)):
			for name in attribute_names:
				Node(name=name, value=self.value[name], parent=self)
		else:
			for name in attribute_names:
				Node(name=name, value=getattr(self.value, name), parent=self)

	def refresh_children(self):
		if len(self.get_indexable_attributes()) != len(self.children):
			self.load_children()

		if isinstance(self.value, (list,dict,np.ndarray,tuple)):
			for child in self.children:
				child.value = self.value[child.name]
				if child.expanded is True:
					child.refresh_children()
		else:
			for child in self.children:
				child.value = getattr(self.value, child.name)
				if child.expanded is True:
					child.refresh_children()


class TreeView(Element):
	def __init__(	self,
					pos, font_size=14,
					parent_node_object=dict(),
					parent_container=None):
		Element.__init__(self=self, parent_container=parent_container)
		self.pos = pos
		self.font = None
		self.font_size = font_size

		self.text_color = c.white
		self.no_setter_text_color = c.grey
		self.active_node_color = (150,50,50)
		self.max_string_length = 80

		self.parent_node_object = parent_node_object # Parent object, from which all branches stem
		self.root = Node(name=type(parent_node_object).__name__, value=parent_node_object, expanded=True)
		self.default_root = self.root
		#self.root.load_children()
		self.selected_node = self.root
		self.pinned_nodes = [] # Nodes to be pinned at the top of the tree view, so you can view/modify them easily

		# Represents the current depth to which each branch (and sub-branch, sub-sub-...) is exploded
		# This may need to keep named references, because the order of properties on an object might change over time? Not sure.
		# Is a simple string list of each displayed item with its corresponding integer depth
		# (contains no actual object references)
		self.current_list = []

	@property
	def font_size(self):
		return self._font_size

	@font_size.setter
	def font_size(self, value):
		self._font_size = value
		self.font = pg.font.Font('Montserrat-regular.ttf', value)

	def _recursed_generate_current_list(self, current_node=None, depth=0):
		if current_node is None:
			current_node = self.root
		if depth == 0:
			self.current_list = []
			for node in self.pinned_nodes:
				self.current_list.append((node, -1))
		if current_node not in self.pinned_nodes:
			self.current_list.append((current_node, depth))
			if current_node.expanded is True:
				current_node.refresh_children()
				for child_node in current_node.children:
					self._recursed_generate_current_list(current_node=child_node, depth=depth+1)

	def _generate_current_list(self, current_node=None, depth=0):
		self._recursed_generate_current_list(current_node=current_node, depth=depth)

		selected_in_current_list = False
		# for depth_pair in self.current_list:
		# 	print(depth_pair[0].name)
		# print('***')

		for depth_node in self.current_list:
			node = depth_node[0]
			# print(node.name, self.selected_node.name)
			# print(node, self.selected_node)
			if node == self.selected_node:
				selected_in_current_list = True
				break

		if selected_in_current_list is False:
			if len(self.current_list) > 0:
				self.selected_node = self.current_list[0][0]


	def any_key_pressed(self, key, mod, unicode_key):
		multiplier = 1
		if mod == pg.KMOD_LSHIFT or mod == pg.KMOD_RSHIFT:
			multiplier = 10

		if str(unicode_key).isalnum() or str(unicode_key) == '_':
			if mod == pg.KMOD_LSHIFT or mod == pg.KMOD_RSHIFT:
				search_list = self.current_list[::-1] # Search list backwards (i.e., seek a node upwards from selected node)
			else:
				search_list = self.current_list

			searching = False
			for node, _ in search_list:
				if searching is True:
					if str(node.name)[0].lower() == str(unicode_key).lower():
						self.selected_node = node
						break
				if node == self.selected_node:
					searching = True

		if key == pg.K_DOWN:
			for i, depth_pair in enumerate(self.current_list):
				node = depth_pair[0]
				depth = depth_pair[1]
				if node == self.selected_node and i < len(self.current_list)-1:
					self.selected_node = self.current_list[i+1][0]
					break
		elif key == pg.K_UP:
			for i, depth_pair in enumerate(self.current_list):
				node = depth_pair[0]
				depth = depth_pair[1]
				if node == self.selected_node and i > 0:
					self.selected_node = self.current_list[i-1][0]
					break
		elif key == pg.K_RETURN:
			if self.selected_node is not None:
				self.selected_node.expanded = not self.selected_node.expanded
				self.selected_node.load_children()
		elif key == pg.K_f and mod == pg.KMOD_LCTRL or mod == pg.KMOD_RCTRL:
			self.root = self.selected_node
		elif key == pg.K_r and mod == pg.KMOD_LCTRL or mod == pg.KMOD_RCTRL:
			self.root = self.default_root
			self.root.expanded = False
			self.selected_node = self.root
		elif key == pg.K_RIGHT:
			value = self.selected_node.value
			if type(value) is bool:
				self.selected_node.set_value(True)
			elif type(value) is int:
					self.selected_node.set_value(value + (1*multiplier))
			elif type(value) is float:
				# Increase by 1%; 10% when boosted (SHIFT);
				# If it's 0.0, set it to 1.0
				if value == 0.0:
					self.selected_node.set_value(1.0)
				else:
					self.selected_node.set_value(value + value*(0.01*multiplier))
			else:
				print("Tried to increment debug value which is not implemented.")
		elif key == pg.K_LEFT:
			value = self.selected_node.value
			if type(value) is bool:
				self.selected_node.set_value(False)
			elif type(value) is int:
				self.selected_node.set_value(value - (1*multiplier))
			elif type(value) is float:
				# Decrease by 1%; 10% when boosted (SHIFT)
				# If it's 0.0, set it to -1.0
				if value == 0.0:
					self.selected_node.set_value(-1.0)
				else:
					self.selected_node.set_value(value - value*(0.01*multiplier))
			else:
				print("Tried to increment debug value which is not implemented.")
		elif key == pg.K_DELETE:
			self.selected_node.set_value(None)
		elif key == pg.K_0:
			value = self.selected_node.value
			if type(value) is int:
				self.selected_node.set_value(0)
			if type(value) is float:
				self.selected_node.set_value(0.0)
		elif key == pg.K_BACKSPACE:
			if self.selected_node.parent is not None:
				if self.root == self.selected_node:
					self.root = self.selected_node.parent
				self.selected_node = self.selected_node.parent
				self.selected_node.expanded = True
		elif key == pg.K_p and mod == pg.KMOD_LCTRL or mod == pg.KMOD_RCTRL:
			if self.selected_node not in self.pinned_nodes:
				# Node isn't pinned, so pin it
				current_list_selected_index = None
				for i, depth_pair in enumerate(self.current_list):
					node = depth_pair[0]
					if node == self.selected_node:
						current_list_selected_index = i


				if current_list_selected_index is None:
					print("selected_node isn't in selected_list. something is probably wrong.")
					return

				self.pinned_nodes.append(self.selected_node)
				if len(self.current_list) <= 1:
					# Our newly pinned node is the only node in the tree, so it will stay selected
					pass
				elif current_list_selected_index == len(self.current_list)-1:
					# Our newly pinned node was the LAST node in current_list, so move the selected node to the now last node
					self.selected_node = self.current_list[current_list_selected_index-1][0]
				else:
					self.selected_node = self.current_list[current_list_selected_index-1][0]

			else:
				# Node is already pinned, so remove the pin
				current_list_selected_index = None
				for i, depth_pair in enumerate(self.current_list):
					if depth_pair[0] == self.selected_node:
						current_list_selected_index = i

				if current_list_selected_index is None:
					print("selected_node isn't in selected_list. something is probably wrong.")
					return

				self.pinned_nodes.remove(self.selected_node)

				if current_list_selected_index == len(self.current_list)-1:
					self.selected_node = self.current_list[current_list_selected_index-1][0]
				elif len(self.current_list) > 1:
					self.selected_node = self.current_list[current_list_selected_index+1][0]
				else:
					self.selected_node = None






	def draw(self, target=draw.screen):
		self.current_list = []
		self._generate_current_list()

		for node in self.pinned_nodes:
			if node.parent is not None:
				node.parent.refresh_children()

		for i, depth_pair in enumerate(self.current_list):
			node = depth_pair[0]
			depth = depth_pair[1]

			if depth == -1: # pinned node:
				string = str(node.name) + ' = ' + str(node.value)
			else:
				string = '>>'*depth + str(node.name) + ' = ' + str(node.value)

			max_string_length = self.max_string_length
			if len(string) > max_string_length:
				string = string[:max_string_length] + ' [...] ' + string[-10:]

			string_surface = self.font.render(string, True, self.text_color)

			if node.parent is not None and self.selected_node is not node:
				try:
					attr = getattr(type(node.parent.value), node.name)
					if isinstance(attr, property):
						if attr.fset == None:
							string_surface = self.font.render(string, True, self.no_setter_text_color)
				except (AttributeError, TypeError):
					pass

			if depth == -1:
				pg.draw.rect(target, c.blue, ((self.pos[0], self.pos[1]+self.font.get_linesize()*i),string_surface.get_size()))
				if node == self.selected_node:
					pg.draw.rect(target, self.active_node_color, ((self.pos[0], self.pos[1]+self.font.get_linesize()*i),string_surface.get_size()), 2)
			elif node == self.selected_node:
				pg.draw.rect(target, self.active_node_color, ((self.pos[0], self.pos[1]+self.font.get_linesize()*i),string_surface.get_size()))
			target.blit(string_surface, (self.pos[0], self.pos[1]+self.font.get_linesize()*i))

class Button(Element):
	def __init__(	self,
					pos, font,
					text,
					align=('left','top'),
					bg_colors={'default': c.black, 'hovered': c.dark_grey, 'pressed': c.green},
					text_colors={'default': c.white, 'hovered': c.white, 'pressed': c.white},
					padding=(10,0),
					parent_container=None):
		Element.__init__(self=self, parent_container=parent_container)
		self.pos = pos
		self.font = font
		self.text = text
		self.padding = padding
		self.align = align
		self.bg_colors = bg_colors
		self.text_colors = text_colors

		self.width = self.font.size(text)[0] + self.padding[0]*2
		self.height = self.font.size(text)[1] + self.padding[1]*2

		self._hovered = False
		self._pressed = False
		self.button_was_pressed = False

		self._generate_surfaces()

	@property
	def size(self):
		return (self.width, self.height)

	@property
	def rect(self):
		offset_x, offset_y = 0, 0
		if self.align[0] == 'center':
			offset_x -= self.width//2
		elif self.align[0] == 'right':
			offset_x -= self.width

		if self.align[1] == 'center':
			offset_y -= self.height//2
		elif self.align[1] == 'down':
			offset_y -= self.height

		return pg.Rect((self.pos[0]+offset_x, self.pos[1]+offset_y), self.size)

	def _generate_surfaces(self):
		self.surfaces = {	'default': pg.Surface(self.size),
							'hovered': pg.Surface(self.size),
							'pressed': pg.Surface(self.size)}

		for key, surface in self.surfaces.items():
			pg.draw.rect(surface, self.bg_colors[key], ((0,0),self.size))
			pg.draw.rect(surface, c.white, ((0,0),self.size),1)
			surface.blit(self.font.render(self.text, True, self.text_colors[key]), self.padding)

	@property
	def hovered(self):
		return self._hovered

	@hovered.setter
	def hovered(self, hovered):
		self._hovered = hovered

	@property
	def pressed(self):
		return self._pressed

	@pressed.setter
	def pressed(self, pressed):
		self._pressed = pressed

	def left_mouse_pressed(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			self.pressed = True
			if self.parent_container:
				self.parent_container.focus_element(self)

	def left_mouse_released(self, mouse_pos):
		if self.pressed == True and self.rect.collidepoint(mouse_pos):
			self.button_was_pressed = True
		self.pressed = False
		if self.parent_container:
			self.parent_container.unfocus_element(self)

	def clear_pressed(self):
		self.button_was_pressed = False

	def update(self, dt, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			self.hovered = True
		else:
			self.hovered = False
			self.pressed = False

	@property
	def state(self):
		_state = 'default'
		if self.pressed:
			_state = 'pressed'
		elif self.hovered:
			_state = 'hovered'

		return _state

	def draw(self, screen):
		draw_offset = draw.draw_surface_aligned(	target=screen,
											source=self.surfaces[self.state],
											pos=self.pos,
											align=self.align)

class TextEntry(Element):
	def __init__(	self,
					pos, font,
					align=('left','top'), text_align=None,
					width=None,
					type='ip',
					label = '',
					text_cursor_scale=0.75, cursor_blink_time=750,
					padding=(5,0),
					alpha=255,
					default_text='',
					parent_container=None):
		Element.__init__(self, parent_container)
		self.pos = pos
		self.width = width
		self.type = type
		self.align = align
		self.text_align = text_align
		self.font = font
		self.alpha = alpha
		self.text_cursor_scale = text_cursor_scale
		self.padding = padding
		self.text = default_text
		self.label = label

		if self.width == None:
			if self.type == 'ip':
				self.width = self.font.size('000.000.000.000')[0] + self.padding[0]*2
			elif self.type == 'port':
				self.width = self.font.size('00000')[0] + self.padding[0]*2
			else:
				self.width = 200

		self.size = (self.width, self.height)
		self.rect = pg.Rect(pos,self.size)

		self.selected_text_indices = None
		self.select_mode = False
		self.select_start_index = None

		self.char_positions = []
		self._calculate_char_positions()

		self.selected = False
		self.cursor_pos = 0
		self.cursor_blink_time = cursor_blink_time
		self.cursor_timer = 0
		self.cursor_visible = True

		self._generate_surfaces()

	@property
	def height(self):
		return self._calculate_height()

	def _calculate_char_positions(self, pos=None):
		char_positions = []
		if pos == None: # Recalculate positions for the whole string
			for i in range(0, len(self.text)+1):
				sub_string = self.text[0:i]
				sub_string_width, _ = self.font.size(sub_string)
				char_positions.append(sub_string_width)
		elif pos >= 0:
			char_positions = self.char_positions[:pos]
			for i in range(pos, len(self.text)+1):
				sub_string = self.text[0:i]
				sub_string_width, _ = self.font.size(sub_string)
				char_positions.append(sub_string_width)

		self.char_positions = char_positions # char_positions[n] is the position of the leftmost pixel of the nth character in the text
		# TODO: Only update the ones past pos for center_positions
		# positon_bounds[n] gives the range for when a click inside the textbox should place the cursor in the nth position (this range is between)
		self.position_bounds = []
		current_pos = 0
		for consecutive_positions in zip(self.char_positions[:], self.char_positions[1:]):
			char_center = (consecutive_positions[0]+consecutive_positions[1])//2 # Finds the center of the nth character
			self.position_bounds.append((current_pos, char_center))
			current_pos = char_center + 1

		self.position_bounds.append((current_pos, self.rect.width))

		# print(self.char_positions)
		# print(list(zip(self.char_positions[:], self.char_positions[1:])))

	def _calculate_height(self):
		test_string = ''
		for i in range(32,127):
			test_string += chr(i) # String containing all 'printable' ASCII characters (that we care about)

		return self.font.size(test_string)[1]

	def _generate_surfaces(self):
		self._generate_box_surface()
		self._generate_label_surface()
		self._generate_text_surface()

	def _generate_box_surface(self):
		self.box_surface = pg.Surface(self.size)

		pg.draw.rect(self.box_surface, c.dark_grey, ((0,0),self.size))
		pg.draw.rect(self.box_surface, c.white, ((0,0),self.size), 1)

		self.box_surface.set_alpha(self.alpha)

	def _generate_label_surface(self):
		self.label_surface = self.font.render(self.label, True, c.grey)
		self.label_surface.set_alpha(self.alpha)

	def _generate_text_surface(self):
		self.text_surface = self.font.render(self.text, True, c.light_grey)
		self.text_selected_surface = self.font.render(self.text, True, c.black)

		self.text_surface.set_alpha(self.alpha)
		self.text_selected_surface.set_alpha(self.alpha)

	@property
	def cursor_pos(self):
		return self._cursor_pos

	@cursor_pos.setter
	def cursor_pos(self, cursor_pos):
		if cursor_pos < 0:
			cursor_pos = 0
		elif cursor_pos > len(self.text):
			cursor_pos = len(self.text)

		self._cursor_pos = cursor_pos
		self.cursor_visible = True
		self.cursor_timer = 0

	def clear_text(self):
		self.text = ''
		self.selected_text_indices = None
		self.select_mode = False
		self.select_start_index = None
		self.cursor_pos = 0
		self.cursor_timer = 0
		self.cursor_visible = True

		self._calculate_char_positions()
		self._generate_surfaces()

	# unselect text and place cursor at cursor_pos
	def _unselect(self, cursor_pos):
		self.cursor_pos = cursor_pos
		self.select_mode = False
		self.selected_text_indices = None
		self.select_start_index = None

	def delete_selected(self):
		left = self.text[:self.selected_text_indices[0]] # left side of selected text
		right = self.text[self.selected_text_indices[1]:] # right ..
		self.text = left + right
		self._unselect(cursor_pos = self.selected_text_indices[0])

	def any_key_pressed(self, key, mod, unicode_key):
		if self.selected == False:
			return

		if key in range(32,127): # a normal 'printable' character
			if self.selected_text_indices != None:
				self.cursor_pos = self.selected_text_indices[0]
				self.delete_selected()

			self.text = self.text[:self.cursor_pos] + unicode_key + self.text[self.cursor_pos:]
			self.cursor_pos += 1
			self._calculate_char_positions(pos = self.cursor_pos-1)
			self._generate_text_surface()

		if key == pg.K_LEFT:
			if self.cursor_pos == 0:
				pass
			elif mod == pg.KMOD_LSHIFT or mod == pg.KMOD_RSHIFT:
				if self.selected_text_indices == None:
					self.selected_text_indices = (self.cursor_pos-1, self.cursor_pos)
				else:
					if self.cursor_pos == self.selected_text_indices[0]:
						self.selected_text_indices = (self.selected_text_indices[0]-1, self.selected_text_indices[1])
					elif self.cursor_pos == self.selected_text_indices[1]:
						self.selected_text_indices = (self.selected_text_indices[0], self.selected_text_indices[1]-1)
					else:
						print("cursor_pos is not equal to either selected_text_index. something went wrong.")

					if self.selected_text_indices[0] == self.selected_text_indices[1]:
						self.selected_text_indices = None
					else:
						self.selected_text_indices = sorted(self.selected_text_indices)

				self.cursor_pos -= 1
			elif self.selected_text_indices != None:
				self._unselect(cursor_pos=self.selected_text_indices[0])
			else:
				self.cursor_pos -= 1
		elif key == pg.K_RIGHT:
			if self.cursor_pos == len(self.text):
				pass
			elif mod == pg.KMOD_LSHIFT or mod == pg.KMOD_RSHIFT:
				if self.selected_text_indices == None:
					self.selected_text_indices = (self.cursor_pos, self.cursor_pos+1)
				else:
					if self.cursor_pos == self.selected_text_indices[0]:
						self.selected_text_indices = (self.selected_text_indices[0]+1, self.selected_text_indices[1])
					elif self.cursor_pos == self.selected_text_indices[1]:
						self.selected_text_indices = (self.selected_text_indices[0], self.selected_text_indices[1]+1)
					else:
						print("cursor_pos is not equal to either selected_text_index. something went wrong.")

					if self.selected_text_indices[0] == self.selected_text_indices[1]:
						self.selected_text_indices = None
					else:
						self.selected_text_indices = sorted(self.selected_text_indices)

				self.cursor_pos += 1
			elif self.selected_text_indices != None:
				self._unselect(cursor_pos=self.selected_text_indices[1])
			else:
				self.cursor_pos += 1
		elif key == pg.K_BACKSPACE:
			if self.selected_text_indices != None:
				self.delete_selected()
			elif self.cursor_pos > 0:
				self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
				self.cursor_pos -= 1

			self._calculate_char_positions(pos=self.cursor_pos)
			self._generate_text_surface()
		elif key == pg.K_DELETE:
			if self.selected_text_indices != None:
				self.delete_selected()
			elif self.cursor_pos < len(self.text):
				self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]

			self._calculate_char_positions(pos=self.cursor_pos)
			self._generate_text_surface()

	# Returns where the cursor should be placed for the given mouse position
	def mouse_pos_to_cursor_index(self, mouse_pos):
		# mouse position relative to the left side of the textbox
		relative_x = mouse_pos[0] - self.rect.left - self.padding[0]

		for i, position_bound in enumerate(self.position_bounds):
			#print('i=%d; position_bound=%s; mouse_pos=%s; relative_x=%s`'%(i, position_bound, mouse_pos, relative_x))
			if i == 0: # valid between -inf up to the second position_bound
				if relative_x <= position_bound[1]:
					return i
			if i == len(self.position_bounds)-1: # valid between first position bound and +inf
				if relative_x >= position_bound[0]:
					return i
			elif relative_x >= position_bound[0] and relative_x <= position_bound[1]:
				return i

		print('mouse_pos_to_cursor_index() failed')
		return 0

	def left_mouse_pressed(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			if self.parent_container:
				self.parent_container.focus_element(self)
			self.selected = True

			new_cursor_pos = self.mouse_pos_to_cursor_index(mouse_pos)
			if new_cursor_pos:
				self.cursor_pos = new_cursor_pos

			self.select_start_index = self.cursor_pos
			self.select_mode = True
			self.cursor_visible = True
			self.cursor_timer = 0
		else:
			if self.parent_container:
				self.parent_container.unfocus_element(self)
			self.selected = False


	def left_mouse_released(self, mouse_pos):
		self.select_mode = False
		self.select_start_index = None


		# if self.selected_text_indices[0] == self.selected_text_indices[1]:
		# 	self.selected_text_indices = None

	def check_mouse_inside(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			return True
		else:
			return False

	def update(self, dt, mouse_pos):
		if self.selected:
			self.cursor_timer += dt
			if self.cursor_timer >= self.cursor_blink_time:
				self.cursor_timer -= self.cursor_blink_time
				self.cursor_visible = not self.cursor_visible

			if self.select_mode == True:
				mouse_index = self.mouse_pos_to_cursor_index(mouse_pos)
				self.cursor_pos = mouse_index
				if self.select_start_index != mouse_index:
					self.selected_text_indices = tuple(sorted([mouse_index, self.select_start_index]))
				else:
					self.selected_text_indices = None


	def _draw_cursor(self, screen):
		if self.cursor_visible:
			x = self.rect.left + self.padding[0] + self.char_positions[self.cursor_pos]
			y_padding = self.rect.height*(1 - self.text_cursor_scale)//2
			pg.draw.line(screen, c.white, (x,self.rect.top+y_padding), (x,self.rect.bottom-y_padding))

	def _draw_text(self, screen):
		# Ignores self.text_align for now
		screen.blit(self.text_surface, (self.rect.left+self.padding[0], self.rect.top+self.padding[1]))
		if self.selected_text_indices != None:
			left_index = self.selected_text_indices[0]
			right_index = self.selected_text_indices[1]
			left = self.char_positions[left_index]
			right = self.char_positions[right_index]
			shifted_left = left + self.rect.left + self.padding[0]
			shifted_right = right + self.rect.left + self.padding[0]

			pg.draw.rect(screen, c.grey, ((shifted_left,self.rect.top),(shifted_right-shifted_left,self.rect.height)))
			screen.blit(self.text_selected_surface, (shifted_left, self.rect.top), (left, 0, right-left, self.text_selected_surface.get_height()))

	def draw(self, screen):
		draw.draw_surface_aligned(	target=screen,
								source=self.box_surface,
								pos=self.pos,
								align=self.align,
								alpha=self.alpha)

		draw.draw_surface_aligned(	target=screen,
								source=self.label_surface,
								pos=self.pos,
								align=('left','down'),
								alpha=self.alpha)

		self._draw_text(screen=screen)

		if self.selected:
			self._draw_cursor(screen=screen)

class ListMenu(Element):
	def __init__(self, items, pos, align, text_align, font, selected_font, item_spacing=4, selected=0, parent_container=None):
		Element.__init__(self, parent_container)
		self.items = items
		self.pos = pos
		self.align = align
		self.text_align = text_align
		self.font = font
		self.selected_font = selected_font
		self.item_spacing = item_spacing
		self.selected = selected
		self.confirmed_index = None

	def _generate_surfaces(self):
		self.item_surfaces = []
		self.selected_item_surfaces = []

		for item in self.items:
			self.item_surfaces.append(self.font.render(item, True, c.light_grey))
			self.selected_item_surfaces.append(self.selected_font.render(item, True, c.gold))

	@property
	def rect(self):
		current_height = 0
		max_width = 0
		for item_index, _ in enumerate(self.items):
			item_surface = self.get_item_surface(item_index)
			current_height += item_surface.get_height()
			max_width = max(max_width, item_surface.get_width())

		return pg.Rect(self.pos, (max_width, current_height))

	@property
	def confirmed_item_text(self):
		if self.confirmed_index != None:
			return self.items[self.confirmed_index]
		else:
			return ''


	@property
	def selected(self):
		return self._selected

	@selected.setter
	def selected(self, selected):
		self._selected = selected
		self._generate_surfaces()

	def clear_confirmed(self):
		self.confirmed_index = None

	def _move_cursor_up(self):
		self.selected -= 1
		if self.selected < 0:
			self.selected = len(self.items)-1

	def _move_cursor_down(self):
		self.selected += 1
		if self.selected >= len(self.items):
			self.selected = 0

	def get_selected_item(self):
		return self.items[self.selected]

	# Menu items can be selected but not hovered. Sometimes, when clicking,
	# you may not want to activate the item unless it's still being hovered
	# (i.e., the mouse is still over the menu element)
	def check_mouse_inside(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			return True
		else:
			return False

	def get_item_surface(self, item_index):
		if item_index == self.selected:
			return self.selected_item_surfaces[item_index]
		else:
			return self.item_surfaces[item_index]

	def any_key_pressed(self, key, mod, unicode_key):
		if key==pg.K_UP or key==pg.K_w or key==pg.K_LEFT or key==pg.K_a:
			self._move_cursor_up()
		elif key==pg.K_DOWN or key==pg.K_s or key==pg.K_RIGHT or key==pg.K_d:
			self._move_cursor_down()
		elif key==pg.K_RETURN or key==pg.K_SPACE:
			self.confirmed_index = self.selected

	def get_hovered_item(self, mouse_pos):
		current_y = 0
		mouse_relative_pos = (mouse_pos[0] - self.pos[0], mouse_pos[1] - self.pos[1])
		for item_index, _ in enumerate(self.items):
			item_surface = self.get_item_surface(item_index)
			# x bounds are already satisfied because of the self.rect.collidepoint() check above
			if mouse_relative_pos[1] >= current_y and mouse_relative_pos[1] < current_y+item_surface.get_height(): # y bounds
				return item_index
			current_y += item_surface.get_height()

	def left_mouse_pressed(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			self.parent_container.focus_element(self)

			hovered = self.get_hovered_item(mouse_pos)
			if hovered != None:
				self.selected = hovered

			self.confirmed_index = self.selected


	def update(self, dt, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			self.parent_container.focus_element(self)

			hovered = self.get_hovered_item(mouse_pos)
			if hovered != None:
				self.selected = hovered


	def draw(self, screen):
		current_y = 0
		for item_index, _ in enumerate(self.items):
			item_surface = self.get_item_surface(item_index)
			screen.blit(item_surface, (self.pos[0], self.pos[0]+current_y))
			current_y += item_surface.get_height()

		# draw.draw_surface_aligned(	target=screen,
		# 						source=self.surface,
		# 						pos=self.pos,
		# 						align=self.align)
		# for item_rect in self.item_rects:
		# 	pg.draw.rect(screen, green, item_rect, 1)

class ChatWindow(Element):
	def __init__(self, name_font, message_font, name_width, message_width, log_height, text_color=c.white, parent_container=None):
		Element.__init__(self, parent_container)
		self.pos = (0,0)
		self.name_font = name_font
		self.message_font = message_font
		self.name_width = name_width
		self.message_width = message_width
		self.log_height = log_height
		self.text_color = text_color
		self.user_color_pool = [c.white, c.grey]
		self.colors_used = 0
		self.user_colors = {"OFFLINE": c.grey}
		self.messages = []

		self.text_entry = TextEntry(pos=(self.pos[0], self.pos[1]+self.log_height),
									font=message_font,
									type='chat',
									width=message_width+name_width,
									alpha=128)

		self.container = Container()
		self.container.add_element(self.text_entry)

	@property
	def width(self):
		return self.name_width + self.message_width

	@property
	def height(self):
		return self.log_height + self.text_entry.height

	@property
	def rect(self):
		return pg.Rect(self.pos, (self.width, self.height))

	def add_message(self, user, text):
		message = (user, text)
		self.messages.append(message)
		if user not in self.user_colors:
			self.user_colors[user] = self.user_color_pool[self.colors_used % len(self.user_color_pool)]
			self.colors_used += 1

	def any_key_pressed(self, key, mode, unicode_key):
		self.container.any_key_pressed(key, mode, unicode_key)
		if key == pg.K_RETURN:
			if len(self.text_entry.text) > 0:
				self.events.append(('send chat message', self.text_entry.text))
				self.text_entry.clear_text()

	def left_mouse_pressed(self, mouse_pos):
		if self.rect.collidepoint(mouse_pos):
			if self.parent_container:
				self.parent_container.focus_element(self)
		else:
			if self.parent_container:
				self.parent_container.unfocus_element()

		self.container.left_mouse_pressed(mouse_pos)

	def left_mouse_released(self, mouse_pos):
		self.container.left_mouse_released(mouse_pos)

	def update(self, dt, mouse_pos):
		self.container.update(dt, mouse_pos)

	def draw(self, screen):
		background_surface = pg.Surface(self.rect.size)
		background_surface.set_alpha(128)
		background_surface.fill(c.dark_grey)
		screen.blit(background_surface, self.pos)

		self.container.draw(screen=screen)

		line_spacing = self.message_font.get_linesize() + 4
		current_line_count = 0

		for message in self.messages[::-1]: # Look through messages backwards, since we only show the most recent ones
			this_line_count = len(util.split_text(text=message[1], font=self.message_font, word_wrap_width=self.message_width))
			current_line_count += this_line_count
			draw.draw_text(
						target=screen,
						text=message[0],
						pos=(self.pos[0], self.pos[1] + self.log_height - current_line_count*line_spacing),
						font=self.name_font,
						color = self.user_colors[message[0]],
						word_wrap = False)
			draw.draw_text(
						target=screen,
						text=message[1],
						pos=(self.name_width + self.pos[0], self.pos[1] + self.log_height - current_line_count*line_spacing),
						font = self.message_font,
						color = util.lighten_color(self.user_colors[message[0]], 0.5),
						word_wrap = True,
						word_wrap_width = self.message_width)
