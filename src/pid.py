class PID:
    def __init__(self, kp, ki, kd, out_low=-1, out_high=1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_low = out_low
        self.out_high = out_high
        self.integral = 0
        self.derivative = 0
        self.prev_error = None

    def reset(self):
        self.integral = 0
        self.prev_error = None
        self.derivative = 0

    def step(self, target, curr, dt):
        error = target - curr
        self.integral += error * dt
        if self.prev_error is None:
            self.derivative = 0
        else:
            self.derivative = (error - self.prev_error) / dt

        self.prev_error = error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * self.derivative)
        output_clamped = max(self.out_low, min(self.out_high, output))

        # anti integral windup
        if output != output_clamped:
            self.integral -= error * dt

        return output_clamped

