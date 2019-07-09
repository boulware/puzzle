from util import last_index, Vec2i

class Tween:
	def __init__(self, start_pos, end_pos, frame_duration, weight=1):
		self.start_pos = start_pos
		self.end_pos = end_pos
		self.frame_duration = frame_duration
		self.current_frame = 0
		self.weight = weight

	@property
	def current_pos(self):
		t = self.current_frame / self.frame_duration
		return ((1-pow(t,weight))*self.start_pos[0] + pow(t,weight)*self.end_pos[0], (1-pow(t,weight))*self.start_pos[1] + pow(t,weight)*self.end_pos[1])

	@property
	def finished(self):
		if self.current_frame >= self.frame_duration:
			return True
		else:
			return False

	def update(self, dt):
		self.current_frame += 1

class EmptyAnimation:
	def __init__(self, pos):
		self.pos = pos

	@property
	def current_pos(self):
		return self.pos

	@property
	def finished(self):
		return False

	def update(self, dt):
		return self.finished

class MoveAnimation:
	def __init__(self, start_pos, end_pos, frame_duration, jerk=1.0):
		self.start_pos = start_pos
		self.end_pos = end_pos
		self.frame_duration = frame_duration
		self.jerk = jerk

		self.current_frame = 0

	@property
	def current_pos(self):
		t = self.current_frame / self.frame_duration
		return ((1-pow(t,self.jerk))*self.start_pos[0] + pow(t,self.jerk)*self.end_pos[0], (1-pow(t,self.jerk))*self.start_pos[1] + pow(t,self.jerk)*self.end_pos[1])

	@property
	def finished(self):
		if self.current_frame >= self.frame_duration:
			return True
		else:
			return False

	def update(self, dt):
		"""Advance animation frame.
		Return True if animation is finished, False otherwise"""
		self.current_frame += 1
		return self.finished

class Sequence:
	"""Contains a sequence of animations that occur one after the other in order"""
	def __init__(self, animations):
		self.animations = animations
		self.current_index = 0

	@property
	def current_animation(self):
		return self.animations[self.current_index]

	@property
	def current_pos(self):
		return self.current_animation.current_pos

	@property
	def finished(self):
		if self.current_index == last_index(self.animations) and self.current_animation.finished is True:
			return True
		else:
			return False

	def update(self, dt):
		if self.finished is True:
			return True

		if self.current_animation.update(dt=dt):
			self.current_index = min(self.current_index + 1, last_index(self.animations))
			if self.current_animation.finished is True:
				return True

		return False

class AttackAnimation(Sequence):
	def __init__(self, start_pos, target_pos, reel_distance=50):
		self.start_pos = Vec2i(tuple=start_pos)
		self.target_pos = Vec2i(tuple=target_pos)


		attack_direction = (self.target_pos - self.start_pos).dir
		reel_target_pos = (self.start_pos.float() - reel_distance*attack_direction).rounded()
		reel_animation = MoveAnimation(start_pos=self.start_pos.as_tuple(), end_pos=reel_target_pos.as_tuple(), frame_duration=20, jerk=0.4)
		hit_animation = MoveAnimation(start_pos=reel_target_pos.as_tuple(), end_pos=self.start_pos.as_tuple(), frame_duration=5)
		bounce_target_pos = (self.start_pos.float() - (reel_distance*0.2)*attack_direction).rounded()
		bounce_animation = MoveAnimation(start_pos=self.start_pos.as_tuple(), end_pos=bounce_target_pos.as_tuple(), frame_duration=4)
		rest_animation = MoveAnimation(start_pos=bounce_target_pos.as_tuple(), end_pos=self.start_pos.as_tuple(), frame_duration=10)

		Sequence.__init__(self=self, animations=[reel_animation,hit_animation,bounce_animation,rest_animation])
