import colors as c
import pygame as pg
import draw
import util

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
		for e in self.elements:
			e.update(dt, mouse_pos)

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

class Button(Element):
	def __init__(	self,
					pos, font,
					text,
					align=('left','top'),
					bg_colors={'default': c.black, 'hovered': c.dark_grey, 'pressed': c.green},
					text_colors={'default': c.white, 'hovered': c.white, 'pressed': c.white},
					padding=(10,0),
					parent_container=None):
		Element.__init__(self, parent_container)
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
			elif mod == pg.KMOD_LSHIFT:
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
			elif mod == pg.KMOD_LSHIFT:
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
		self.user_colors = {"Offline": c.grey}
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
			draw.draw_text(	text=message[0],
						pos=(self.pos[0], self.pos[1] + self.log_height - current_line_count*line_spacing),
						font=self.name_font,
						color = self.user_colors[message[0]],
						word_wrap = False)
			draw.draw_text(	text=message[1],
						pos=(self.name_width + self.pos[0], self.pos[1] + self.log_height - current_line_count*line_spacing),
						font = self.message_font,
						color = lighten_color(self.user_colors[message[0]], 0.5),
						word_wrap = True,
						word_wrap_width = self.message_width)
