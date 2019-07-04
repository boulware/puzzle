import pygame as pg

import constants as c
import debug as d
import util

screen = None

#@d.info
def draw_surface_aligned(target, source, pos, align=('left','left'), offset=(0,0), alpha=255):
	align_offset = list(offset)

	if align[0] == 'center':
		align_offset[0] -= source.get_width()//2
	elif align[0] == 'right':
		align_offset[0] -= source.get_width()

	if align[1] == 'center':
		align_offset[1] -= source.get_height()//2
	elif align[1] == 'down':
		align_offset[1] -= source.get_height()

	new_x = pos[0] + align_offset[0]
	new_y = pos[1] + align_offset[1]
	new_pos = (new_x, new_y)

	target.blit(source, new_pos)

	return align_offset

def draw_text(target, text, pos, font, color=c.white, word_wrap=False, word_wrap_width=None):
	if len(text) == 0:
		line_count = 0
	if word_wrap == False:
		text_surface = pg.Surface(font.size(text))
		text_surface.set_colorkey(c.black)
		text_surface.fill(c.black)
		text_surface.blit(font.render(text, True, color), (0,0))
		line_count = 1
	else:
		line_spacing = font.get_linesize()
		lines = util.split_text(text, font, word_wrap_width=word_wrap_width)
		line_count = len(lines)
		text_surface = pg.Surface((word_wrap_width, line_spacing*line_count))
		text_surface.set_colorkey(c.black)
		text_surface.fill(c.black)

		for line_number, line in enumerate(lines):
			text_surface.blit(font.render(line, True, color), (0,line_number*line_spacing))

	target.blit(text_surface, pos)

	return line_count