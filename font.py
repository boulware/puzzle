import pygame as pg

class Fonts:
	def load(self):
		self.card_text_v_sm = pg.font.Font("Montserrat-Regular.ttf", 12)
		self.card_text_sm = pg.font.Font("Montserrat-Regular.ttf", 18)
		self.card_text_med = pg.font.Font("Montserrat-Regular.ttf", 24)
		self.card_text_lg = pg.font.Font("Montserrat-Regular.ttf", 32)
		self.node_font = pg.font.Font("Montserrat-Regular.ttf", 26)
		self.count_font = pg.font.Font("Montserrat-Regular.ttf", 14)
		self.ui_font = pg.font.Font("Montserrat-Regular.ttf", 24)
		self.action_font = pg.font.Font("Montserrat-Regular.ttf", 12)
		self.main_menu_font = pg.font.Font("Montserrat-Regular.ttf", 48)
		self.main_menu_font_med = pg.font.Font("Montserrat-Regular.ttf", 32)
		self.main_menu_font_small = pg.font.Font("Montserrat-Regular.ttf", 18)
		self.chat_message_font = pg.font.Font("Montserrat-Regular.ttf", 16)
		self.chat_name_font = pg.font.Font("Montserrat-Regular.ttf", 16)
		self.main_menu_selected_font = pg.font.Font("Montserrat-Regular.ttf", 60)

fonts = Fonts()