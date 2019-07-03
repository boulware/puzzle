import traceback
print_callstack = traceback.print_stack

def split_text(text, font, word_wrap_width):
		lines = []

		split_text = text.split(' ')
		current_line = ''
		for i, word in enumerate(split_text):
			if i == 0:
				line_width = font.size(word)[0]
			else:
				line_width = font.size(current_line + ' ' + word)[0]
			if line_width >= word_wrap_width:
				lines.append(current_line)
				current_line = word
			else:
				if i == 0:
					current_line += word
				else:
					current_line += ' ' + word

		if len(current_line) > 0:
			lines.append(current_line)

		return lines

def clamp(value, min_value, max_value):
	clamped_value = value
	if value < min_value:
		clamped_value = min_value
	elif value > max_value:
		clamped_value = max_value

	return clamped_value