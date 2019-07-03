import util

# General constants
black = (0,0,0)
grey = (127,127,127)
light_grey = (200,200,200)
dark_grey = (40,40,40)
white = (255,255,255)
red = (255,0,0)
light_red = (255,100,100)
dark_red = (70,0,0)
very_dark_red = (40,0,0)
green = (0,255,0)
light_green = (0,150,0)
dark_green = (0,70,0)
very_dark_green = (0,40,0)
blue = (0,0,255)
light_blue = (100,100,255)
dark_blue = (0,0,70)
very_dark_blue = (0,0,40)
purple = (128,0,128)
gold = (255,215,0)
pink = (255,200,200)

# Scales color towards (0,0,0), where amount is between 0 and 1 (1 taking it all the way to c.black)
def darken_color(color, amount):
	try:
		new_color = list(color)
		for i, channel in enumerate(new_color):
			new_color[i] = util.clamp(channel*(1-amount), 0, 255)
		return new_color
	except IndexError:
		print("Failed to darken color.")
		return color

# Scales color towards (0,0,0), where amount is between 0 and 1 (1 takes it all the way to c.white)
def lighten_color(color, amount):
	try:
		new_color = list(color)
		for i, channel in enumerate(new_color):
			new_color[i] = util.clamp(channel + amount*(255-channel), 0, 255)
		return new_color
	except:
		print("Failed to darken color.")
		return color