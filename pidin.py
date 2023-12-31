import djitellopy as tello
import time
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

me.takeoff()

desired_height = -3  # in centimeters
# me.send_rc_control(0, 0, desired_height, 0)

class PIDController2D:
    def __init__(self, kp_x, ki_x, kd_x, kp_y, ki_y, kd_y):
        # Initialize PID controller with coefficients for X and Y axes
        self.kp_x = kp_x  # Proportional gain for X axis
        self.ki_x = ki_x  # Integral gain for X axis
        self.kd_x = kd_x  # Derivative gain for X axis
        self.kp_y = kp_y  # Proportional gain for Y axis
        self.ki_y = ki_y  # Integral gain for Y axis
        self.kd_y = kd_y  # Derivative gain for Y axis

        # Initialize error and integral terms for both axes
        self.prev_error_x = 0
        self.integral_x = 0
        self.prev_error_y = 0
        self.integral_y = 0

    def control(self, setpoint_x, current_position_x, setpoint_y, current_position_y):
        # Calculate error for X and Y axes
        error_x = setpoint_x - current_position_x
        error_y = setpoint_y - current_position_y

        # Update integral terms for both axes
        self.integral_x += error_x
        self.integral_y += error_y

        # Calculate derivative terms for both axes
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y

        # Calculate control outputs for X and Y axes using PID equations
        output_x = self.kp_x * error_x + self.ki_x * self.integral_x + self.kd_x * derivative_x
        output_y = self.kp_y * error_y + self.ki_y * self.integral_y + self.kd_y * derivative_y

        # Update previous error terms for both axes
        self.prev_error_x = error_x
        self.prev_error_y = error_y

        # Return control outputs for both axes
        return output_x, output_y

# Simulated 2D system
class System2D:
    def __init__(self, initial_x, initial_y):
        # Initialize the system with initial positions for X and Y
        self.x = initial_x
        self.y = initial_y

    def update(self, control_output_x, control_output_y):
        # Simulate the system's response to the control outputs
        self.x += control_output_x
        
        self.y += control_output_y

# Main loop
if __name__ == "__main__":
    # Initialize the PID controller, system, setpoints, and time steps
    pid_2d = PIDController2D(kp_x=0.9, ki_x=0.1, kd_x=0.2, kp_y=0.45, ki_y=0.1, kd_y=0.45)
    system_2d = System2D(initial_x=0, initial_y=0)
    puerta=int(input("Escribe que puerta quieres (1,2,3,4)"))
    setpoint_x = 0
    setpoint_y = 0
    # me.TIME_BTW_RC_CONTROL_COMMANDS=1000
    if puerta == 1:
        setpoint_x=20
        time.sleep(1)
        me.send_rc_control(20, 10, 0, 50)
        time.sleep(10)
        me.send_rc_control(0, 0, 0, 0)
        setpoint_y=0
    elif puerta ==2:
        setpoint_x=20
        me.move_right(35)
        setpoint_y = 0
    elif puerta ==3:
        setpoint_x= 10
        me.move_right(20)
        me.move_up(20)
        setpoint_y = 5
    elif puerta == 4:
        setpoint_x= 10
        me.move_right(20)
        me.move_down(20)
        setpoint_y = -5

    time_steps = 100
    

    for _ in range(time_steps):
        # Get current positions for X and Y
        current_position_x = system_2d.x
        current_position_y = system_2d.y

        # Calculate control outputs for both axes using the PID controller
        control_output_x, control_output_y = pid_2d.control(setpoint_x, current_position_x, setpoint_y, current_position_y)
        # if control_output_x < 0 :
        #     me.move_left(int(control_output_x*-1))
        # elif control_output_x> 0 :
        #     me.move_right(int(control_output_x))
        # # Update the system with the control outputs
        # system_2d.update(control_output_x, control_output_y)

        #holgura
        wiggle=0.5

        if setpoint_x-wiggle <=current_position_x <= setpoint_x+wiggle and setpoint_y-wiggle <=current_position_y <= setpoint_y+wiggle:
            print("AVANZA AVANZA AVANZAAAAAA")
            break

        # Print information for both X and Y axes
        #print(f"Setpoint X: {setpoint_x}, Current Position X: {current_position_x}, Control Output X: {control_output_x}")
        #print(f"Setpoint Y: {setpoint_y}, Current Position Y: {current_position_y}, Control Output Y: {control_output_y}")

