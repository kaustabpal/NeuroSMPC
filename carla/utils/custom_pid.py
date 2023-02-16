import utils.PID as simple_pid


class PID:
    def __init__(self, Kp, Ki, Kd, SP, sample_time=0.01, error_mag_integral = None, output_limits = (None, None), name='base_pid'):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.SP = SP
        self.sample_time = sample_time  # Time after which PID value will change
        self.pid_instance = None
        self.output_limits = output_limits
        self.error_mag_integral = error_mag_integral
        self.create_pid_instance()
        self.name = name

    def create_pid_instance(self):
        self.pid_instance = simple_pid.PID(
            Kp=self.Kp, Ki=self.Ki, Kd=self.Kd, setpoint=self.SP, sample_time=self.sample_time, error_mag_integral = self.error_mag_integral)
        self.pid_instance.output_limits = self.output_limits # (-0.25, 0.25)

    def get_output(self, SP, MV, dt):
        if self.SP != SP:
            print('[INFO] ' + self.name + ': Set point changed to '+ str(SP) +', recreating pid instance')
            self.SP = SP
            self.create_pid_instance()
        return self.pid_instance(MV, dt=dt)
