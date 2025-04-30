# -*- coding: utf-8 -*-

"""Version 7.3 FLight time calculations for UAV in 4-D dimension."""
# NEU coordinate system.
# We use an assumption that UAV max thrust is known.
"""Updates:
- Fixed bug with max thrust fro flight mode IA.
- Rewritten method for finding v_max.
"""

import numpy as np
import logging

# Class definition for UAV flight model
class UAVFlightModel:
    initial_velocities = []  # List of segment initial velocities (v_0 for Parent waypoints)
    v_list = []  # List of the velocities (final velocity at approaching Child waypoints)


    # Initialize UAV flight model with the provided initial conditions
    def __init__(self, m, f_t, csa, c_d, g, v_0, rho, theta_limit, thrust_min_percent, delta_time):
        self.m = m  # Mass of the UAV, kg
        self.f_t = f_t  # Maximum thrust force, N
        self.csa = csa  # Cross-sectional area, m^2
        self.c_d = c_d  # Drag coefficient
        self.g = g  # Acceleration due to gravity, m/s^2
        self.rho = rho  # Air density, kg/m^3
        self.theta_limit = np.deg2rad(theta_limit)  # Maximum tilt angle, radians
        self.gamma_theta_limit = None  # Trajectory angle gamma at gamma limit and maximum thrust
        
        self.w_x = None  # Waypoint 2 x-coordinate in body-fixed coordinate system, m
        self.w_z = None  # Waypoint 2 z-coordinate in body-fixed coordinate system, m
        self.epsilon = None  # Angle of flight path change, degrees
        self.thrust_min_percent = thrust_min_percent  # Minimum thrust percentage
        self.delta_time = delta_time  # Time step for Euler's method, s
        self.b = 0.5 * self.rho * self.csa * self.c_d  # "b" is a constant to simplify the calculations
        self.distance_plnd = None  # Distance of Paren-Child segment, m
        self.gamma = None  # Trajectory angle gamma
        self.t_final = 0  # Final time, s
        self.time_simple = None  # Time in the simple model, s
        self.v_0 = v_0  # Initial velocity for the first vector calculations, m/s
        self.v = v_0  # Velocity during flight, m/s
        self.theta = None  # Tilt angle gamma
        self.theta_1 = None
        self.theta_2 = None
        self.delta = None  # Z-component of the f_tr force
        self.delta_1 = None
        self.delta_2 = None
        self.f_x = None  # X-component of the f_tr force
        self.f_x_theta_limit = None  # X-component of the f_tr force at gamma limit
        self.f_tr = None  # Net force without drag impact
        self.f_x_red = None  # X-component of the f_tr force with decreased thrust
        self.f_tr_red = None  # Net force without drag impact with decreased thrust
        self.f_t_red = None  # Thrust force with decreased thrust
        self.f_t_change = None  # Maximum thrust change

        self.v_max = self.calculatiuons_for_v_max() # Maximum velocity in the simple model, m/s

    def calculatiuons_for_v_max(self):

        gamma = 0
        f_t = self.f_t

        # Finding gamma_at_theta_limit
        f_x_theta_limit = np.sin(self.theta_limit) * f_t
        delta_theta_limit = np.cos(self.theta_limit) * f_t - self.m * self.g
        f_tr_theta_limit = np.sqrt(delta_theta_limit ** 2 + f_x_theta_limit ** 2)
        gamma_theta_limit = np.arcsin((np.cos(self.theta_limit) * f_t - (self.m * self.g))
                                           / f_tr_theta_limit)  # Angle gamma at theta limit, radians

        # Finding the tilt angle Theta for the maximum thrust
        # Finding two deltas via discriminant
        p = 2 * ((np.sin(gamma)) ** 2) * self.m * self.g
        c = ((np.sin(gamma)) ** 2) * ((self.m * self.g) ** 2 - f_t ** 2)
        discriminant = p ** 2 - 4 * c
        delta_1 = round((-p + np.sqrt(discriminant)) / 2, 4)

        # Choosing proper delta based on the physical sense
        theta_1 = round(np.arccos((delta_1 + self.m * self.g) / f_t), 4)  # Radians

        # Flight mode IB calculations for finding V_max
        if np.rad2deg(gamma_theta_limit) < 0:
            delta = delta_1  # Radians
            theta = theta_1  # Radians
            f_x = round(np.sin(theta) * f_t, 4)
            f_tr = round(((delta) ** 2 + (f_x) ** 2) ** 0.5, 4)

        # Flight Mode IIA calculations for finding V_max
        if np.rad2deg(gamma_theta_limit) >= 0:
            # Calculations for the reduced thrust
            delta = 0
            f_x_red = np.tan(self.theta_limit) * (self.m * self.g + delta)
            f_tr_red = (delta ** 2 + f_x_red ** 2) ** 0.5
            f_tr = f_tr_red  # Net force without drag impact with decreased thrust

        # Initialize variables
        dist_trvld = 0
        vv = 0  # Speed value for finding v_max
        # Iterate using Euler's method
        while True:
            # Calculate net force
            f_net = f_tr - 0.5 * self.rho * self.csa * self.c_d * vv ** 2
            # Calculate acceleration
            a = f_net / self.m
            # Update velocity and position
            vv += a * self.delta_time  # Velocity value for finding v_max
            dist_trvld += vv * self.delta_time
            # Check for termination condition (desired distance_plnd reached)
            if round(a, 4) == 0:
                return  round(vv, 1)


    def modelling_inertia(self,v_f_parent):
        if v_f_parent == None:
            # Initial velocity for Parent-Child vector calculations for the first waypoint, m/s:
            self.v_0 = 0

        else:
            # Points in 3D space
            GP = (self.gp_N, self.gp_E, self.gp_U)
            Par = (self.prnt_N, self.prnt_E, self.prnt_U)
            Chld = (self.chld_N, self.chld_E, self.chld_U)

            # Compute vectors from points
            GP_Par = (Par[0] - GP[0], Par[1] - GP[1], Par[2] - GP[2])
            Par_Chld = (Chld[0] - Par[0], Chld[1] - Par[1], Chld[2] - Par[2])

            # Dot product
            dot_product = GP_Par[0] * Par_Chld[0] + GP_Par[1] * Par_Chld[1] + GP_Par[2] * Par_Chld[2]

            # Magnitudes
            magnitude_GP_Par = np.sqrt(GP_Par[0] ** 2 + GP_Par[1] ** 2 + GP_Par[2] ** 2)
            magnitude_Par_Chld = np.sqrt(Par_Chld[0] ** 2 + Par_Chld[1] ** 2 + Par_Chld[2] ** 2)

            # Angle epsilon in radians
            cos_epsilon = dot_product / (magnitude_GP_Par * magnitude_Par_Chld)
            if cos_epsilon < -1 or cos_epsilon > 1:
                cos_epsilon = round(cos_epsilon)
            self.epsilon = np.arccos(cos_epsilon)

            # Initial velocity for Parent-Child vector calculations, m/s:
            self.v_0 = round(np.cos(self.epsilon) * v_f_parent, 1)

        # return initial velocity
        return self.v_0 


    # Convert the NEU coordinates to a body-fixed coordinate system where the Parent waypoint is at the origin.
    def transform_to_vertical_plane(self):
        WPpar = np.array([self.prnt_N, self.prnt_E, self.prnt_U])
        WPchld = np.array([self.chld_N, self.chld_E, self.chld_U])

        if self.prnt_N == self.chld_N and self.prnt_E == self.chld_E and self.prnt_U == self.chld_U:
            print("Parent Waypoint and Child Waypoint are the same. No need to change UAV position.")
            exit()
        else:
            WParr_prime = WPchld - WPpar

            # z-axis is the vertical axis
            z_axis = np.array([0, 0, 1])

            # x-axis is the horizontal component of the vector from the Parent to the Child waypoint
            horizontal_WPdepWParr = np.array([WParr_prime[0], WParr_prime[1], 0])
            if np.linalg.norm(horizontal_WPdepWParr) == 0:
                x_axis = np.array([1, 0, 0])  # Default x-axis if no horizontal difference
            else:
                x_axis = horizontal_WPdepWParr / np.linalg.norm(horizontal_WPdepWParr)

            # y-axis is the cross product of z and x axes
            y_axis = np.cross(z_axis, x_axis)

            # Transform point to new basis
            transformed_point = np.array([
                np.dot(WPchld - WPpar, x_axis),
                np.dot(WPchld - WPpar, y_axis),
                np.dot(WPchld - WPpar, z_axis)
            ])

            self.w_x = round(transformed_point[0], 1)  # Child waypoint x-coordinate in body-fixed coordinate system, m
            self.w_z = round(transformed_point[2], 1)  # Child waypoint z-coordinate in body-fixed coordinate system, m
            self.distance_plnd = np.sqrt(self.w_x ** 2 + self.w_z ** 2)  # Distance to the waypoint, m


    # Check that waypoint w_x-coordinate must be greater than or equal to 0
    def check_waypoint_coordinates(self):
        if self.w_x < 0:
            print("Input Error. Waypoint w_x-coordinate must be greater than or equal to 0!")
            exit()
        if self.w_x == 0 and self.w_z == 0:
            print("Waypoint is at the origin. No need to change UAV position.")
            print(self.prnt_N, self.prnt_E, self.prnt_U)
            print(self.chld_N, self.chld_E, self.chld_U)
            exit()


    # Calculate the angle gamma in radians
    def calculate_gamma(self):
        if self.w_x > 0:
            self.gamma = np.arctan(self.w_z / self.w_x)
        elif self.w_x == 0 and self.w_z > 0:
            self.gamma = np.pi / 2
        elif self.w_x == 0 and self.w_z < 0:
            self.gamma = -np.pi / 2


    # Finding Gamma at Theta_limit for the maximum thrust
    def gamma_at_theta_limit(self):
        self.f_x_theta_limit = np.sin(self.theta_limit) * self.f_t
        delta_theta_limit = np.cos(self.theta_limit) * self.f_t - self.m * self.g
        f_tr_theta_limit = np.sqrt(delta_theta_limit ** 2 + self.f_x_theta_limit ** 2)
        self.gamma_theta_limit = np.arcsin((np.cos(self.theta_limit) * self.f_t - (self.m * self.g))
                                           / f_tr_theta_limit)  # Angle gamma at theta limit, radians


    # Finding the tilt angle Theta for the maximum thrust
    def theta_calculations(self):
        # if/else is required as we deal with arccos function
        # Condition for non-vertical flight
        if self.w_x != 0:
            # Finding two deltas via discriminant
            p = 2 * ((np.sin(self.gamma)) ** 2) * self.m * self.g
            c = ((np.sin(self.gamma)) ** 2) * ((self.m * self.g) ** 2 - self.f_t ** 2)
            discriminant = p ** 2 - 4 * c
            self.delta_1 = round((-p + np.sqrt(discriminant)) / 2, 4)
            self.delta_2 = round((-p - np.sqrt(discriminant)) / 2, 4)

            # Choosing proper delta based on the physical sense
            self.theta_1 = round(np.arccos((self.delta_1 + self.m * self.g) / self.f_t), 4)  # Radians
            self.theta_2 = round(np.arccos((self.delta_2 + self.m * self.g) / self.f_t), 4)  # Radians
        # Condition for vertical flight
        else:
            self.theta = 0  # Tilt angle theta, radians


    # Finding forces and angles based on the flight modes.
        # Flight Modes "A" - UAV with a high thrust-to-weight ratio (gamma_theta_limit >= 0).
        # Flight Modes "B" - UAV with a low thrust-to-weight ratio (gamma_theta_limit < 0).
    def flight_modes_IA_IB_IIB(self):
        def finding_fx_ftr():
            self.f_x = round(np.sin(self.theta) * self.f_t, 4)
            self.f_tr = round(((self.delta) ** 2 + (self.f_x) ** 2) ** 0.5, 4)

        # Flight Mode IA. Ascend with gamma > 0, Thrust maximum.
        # Flight Mode IB. Ascend (or horizontal flight) with positive gamma < 90 degrees, Thrust maximum.
        if self.in_flight_mode_IA() or self.in_flight_mode_IB():
            if np.rad2deg(self.gamma_theta_limit) >= 0:
                logging.debug("Flight Mode: IA")
            else:
                logging.debug("Flight Mode: IB")
            self.delta = self.delta_1  # Radians
            self.theta = self.theta_1  # Radians
            finding_fx_ftr()  # Call the function to calculate f_x and f_tr

        # Flight Mode IIB. Descend with a small angle, Thrust maximum.
        elif self.in_flight_mode_IIB():
            logging.debug("Flight Mode: IIB")
            self.delta = self.delta_2
            self.theta = self.theta_2
            finding_fx_ftr()  # Call the function to calculate f_x and f_tr


    def in_flight_mode_IA(self):
        return np.rad2deg(self.gamma) >= np.rad2deg(self.gamma_theta_limit) \
            and np.rad2deg(self.gamma) < 90 \
            and np.rad2deg(self.gamma_theta_limit) >= 0 \
            and self.w_x != 0

    def in_flight_mode_IB(self):
        return np.rad2deg(self.gamma) >= 0 \
            and np.rad2deg(self.gamma) < 90 \
            and np.rad2deg(self.gamma_theta_limit) < 0 \
            and self.w_x != 0

    def in_flight_mode_IIB(self):
        return np.rad2deg(self.gamma_theta_limit) < 0 \
            and np.rad2deg(self.gamma) < 0 \
            and np.rad2deg(self.gamma) >= np.rad2deg(self.gamma_theta_limit) \
            and self.w_x != 0

    def flight_modes_IIA_IIIA_IIIB(self):
        if np.rad2deg(self.gamma) < np.rad2deg(self.gamma_theta_limit):
            # Flight Modes IIA, IIIA, IIIB calculations and corresponded calculations
            # Calculations for the reduced thrust
            if np.rad2deg(self.gamma) != 0:
                self.delta = (np.tan(self.gamma) * np.tan(self.theta_limit) * self.m * self.g /
                             (1 - np.tan(self.gamma) * np.tan(self.theta_limit)))
            elif np.rad2deg(self.gamma) == 0:
                self.delta = 0
            self.f_x_red = np.tan(self.theta_limit) * (self.m * self.g + self.delta)
            self.f_tr_red = (self.delta ** 2 + self.f_x_red ** 2) ** 0.5
            self.f_t_red = (self.f_x_red ** 2 + (self.m * self.g + self.delta) ** 2) ** 0.5
            self.f_t_change = 100 * self.f_t_red / self.f_t  # Change of thrust
            self.f_tr = self.f_tr_red  # Net force without drag impact with decreased thrust
            self.theta = self.theta_limit  # Tilt angle theta, radians

    def in_flight_mode_IIA(self):
        return np.rad2deg(self.gamma_theta_limit) >= 0 \
            and np.rad2deg(self.gamma) < np.rad2deg(self.gamma_theta_limit) \
            and np.rad2deg(self.gamma) >= 0 \
            and self.w_x != 0
    def in_flight_mode_IIIA(self):
        return np.rad2deg(self.gamma) < 0 \
            and np.rad2deg(self.gamma_theta_limit) >= 0 \
            and np.rad2deg(self.gamma) > -90 \
            and self.w_x != 0

    def in_flight_mode_IIIB(self):
        return np.rad2deg(self.gamma) < np.rad2deg(self.gamma_theta_limit) \
            and np.rad2deg(self.gamma_theta_limit) < 0 \
            and np.rad2deg(self.gamma) > -90 \
            and self.w_x != 0

    # Vertical ascend and decend flight modes
    def vertical_flight_modes(self):
        # Vertical ascend.
        if self.w_x == 0 and self.w_z > 0:
            self.f_tr = self.f_t - self.m * self.g

        # Flight mode Vertical descend.
        elif self.w_x == 0 and self.w_z < 0:
            self.f_t_red = self.f_t * self.thrust_min_percent / 100  # Tmax decreased for vertical descend
            self.f_tr = self.m * self.g - self.f_t_red
            self.f_t_change = round(100 * self.f_t_red / self.f_t, 2)
            self.theta = 0  # Tilt angle theta, radians
            self.delta = -1 * self.f_tr

    
    def printing_flight_modes(self):
        if self.in_flight_mode_IA() or self.in_flight_mode_IB():
            if np.rad2deg(self.gamma_theta_limit) >= 0:
                print("Flight Mode: IA")
            else:
                print("Flight Mode: IB")

        if self.in_flight_mode_IIB():
            print("Flight Mode: IIB")

        # Flight Mode IIA. Ascend with a small angle gamma, Thrust decreased.
        if self.in_flight_mode_IIA():
            print("Flight Mode: IIA")

        # Flight Mode IIIA. Descend with gamma < 0, Thrust decreased.
        # Flight Mode IIIB. Descend with gamma < gamma_theta_limit, Thrust decreased.
        elif self.in_flight_mode_IIIA() or self.in_flight_mode_IIIB():
            if np.rad2deg(self.gamma_theta_limit) >= 0:
                print("Flight Mode: IIIA")
            else:
                print("Flight Mode: IIIB")

        # Vertical ascend.
        if self.w_x == 0 and self.w_z > 0:
            print("Flight Mode: Vertical ascend")

        # Flight mode Vertical descend.
        elif self.w_x == 0 and self.w_z < 0:
            print("Flight Mode: Vertical descend")


    # Finding the final time and distance traveled via Euler's method
    def finding_time_distance_Euler(self):
        # Initialize variables
        t = 0
        dist_trvld = 0
        # Iterate using Euler's method
        while True:
            # Calculate net force
            f_net = self.f_tr - 0.5 * self.rho * self.csa * self.c_d * self.v ** 2
            # Calculate acceleration
            a = f_net / self.m
            # Update velocity and position
            # print("velocity:", self.v, "acceleration:", a, "time:", t, "Distance traveled (m):", dist_trvld)
            self.v += a * self.delta_time
            dist_trvld += self.v * self.delta_time
            # Update time
            t += self.delta_time

            # Check for termination condition (desired distance_plnd reached)
            if dist_trvld >= self.distance_plnd:
                self.t_final = t
                break
        # append the final velocity to the list
        return round(self.v, 1)
    
    # Finding the final time and distance traveled via Euler's method and simple calculations
    # once terminal velocity has been reached
    def finding_time_distance_Euler_optimised(self):
        # Initialize variables
        t = 0
        dist_trvld = 0
        # Tolerance for acceleration (m/s^2) to make sure that terminal velocity has been reached
        tol = 0.01
        # Iterate using Euler's method
        while True:
            # Update time
            t += self.delta_time
            # Calculate net force
            f_net = self.f_tr - 0.5 * self.rho * self.csa * self.c_d * self.v ** 2
            # Calculate acceleration
            a = f_net / self.m
            # condition for calculating velocity until the terminal one has been reached
            if abs(a) > tol:
                # Update velocity
                self.v += a * self.delta_time
                # Update position
                dist_trvld += self.v * self.delta_time
            # condition for calculating velocity once the terminal one has been reached
            if abs(a) <= tol:
                #print("Tolerance for acceleration:", tol)
                #print("Flight with acceleration != 0. Distance (m):", round(dist_trvld,2), "Time (s):", round(t, 2))
                remaining_dist = self.distance_plnd - dist_trvld  # distance of flight with terminal velocity
                t_with_vt = remaining_dist / self.v  # time of flight with terminal velocity
                t = t + t_with_vt  # update the total time
                dist_trvld = self.distance_plnd  # Update position
                #print(f"Total values. v (m/s): {self.v:.1f}, a (m/s^2): {a:.2f}, t (s): {t:.2f}, Dist_trvld (m): {dist_trvld:.2f}")
                break
            # Check for termination condition (desired distance_plnd reached)
            if dist_trvld >= self.distance_plnd:
                self.t_final = t
                break

        return round(self.v, 1)


    def finding_time_distance_simple(self):
        # Initialize variables
        self.time_simple = self.distance_plnd / self.v_max  # Time in the simple model, s
        return self.time_simple   


    # Main function to simulate the flight
    def simulate_flight(self,v_f_parent,waypoints, is_log=False):
        # Unpack waypoints
        self.gp_N, self.gp_E, self.gp_U = waypoints[0]  # Grandparent waypoint coordinates
        self.prnt_N, self.prnt_E, self.prnt_U = waypoints[1]  # Parent waypoint coordinates
        self.chld_N, self.chld_E, self.chld_U = waypoints[2]  # Child waypoint coordinates

        v_0 = self.modelling_inertia(v_f_parent)
        self.transform_to_vertical_plane()
        self.check_waypoint_coordinates()
        self.calculate_gamma()
        self.gamma_at_theta_limit()
        self.theta_calculations()
        self.flight_modes_IA_IB_IIB()
        self.flight_modes_IIA_IIIA_IIIB()
        self.vertical_flight_modes()
        #v_f = self.finding_time_distance_Euler()
        v_f = self.finding_time_distance_Euler_optimised()
        #print('found euler velocity:', v_f)
        self.finding_time_distance_simple()

        # Printing final results
        if is_log:
            #if i > 0:
            print(f"Coordinates of WP{0}, in NEU system, m. N: {self.gp_N}, E: {self.gp_E}, U: {self.gp_U}")
            print(f"Coordinates of WP{1}, in NEU system, m. N: {self.prnt_N}, E: {self.prnt_E}, U: {self.prnt_U}")
            print(f"Coordinates of WP{2}, in NEU system, m. N: {self.chld_N}, E: {self.chld_E}, U: {self.chld_U}")
            print("------------------------")
            print(f"  Coordinates of WP{2} in body-fixed system, m. w_x: {self.w_x}, w_z: {self.w_z}")

            if self.f_t_change is None:
                print("Maximum thrust")  # Maximum thrust
            else:
                print(f"Thrust reduced to, %: {self.f_t_change:.1f}")
            print(f"  Limit for tilt angle gamma, degrees: {np.rad2deg(self.theta_limit):.1f}")
            print(f"  Angle gamma, degrees: {np.rad2deg(self.gamma):.1f}")
            print(f"  Angle theta, degrees: {np.rad2deg(self.theta):.1f}")

            print("  Gamma at Theta_limit, degrees:", round(np.rad2deg(self.gamma_theta_limit), 2))
            print("------------------------")
            print(f"Distance between WP{1} and WP{2}, m: {self.distance_plnd:.1f}")
            print(f"Flight time to fly between WP{1} and WP{2}, s: {self.t_final:.1f}")
            print(f"Flight time in SIMPLE MODEL based on V_max ({self.v_max} m/s): {self.time_simple:.1f}, s")
            print(f"Initial velocity at WP{1}, m/s: {self.v_0:.1f}")
            print(f"Final velocity at WP{2}, m/s: {self.v:.1f}")
            print("==================================================")
        return v_0, v_f, self.t_final, self.time_simple

if __name__  == '__main__':
        
    # Define waypoints in NEU reference system
    list_waypoints = [
            (500, 500, 0),   # Waypoint 1 (prnt_N, prnt_E, prnt_U)
            (1000, 1000, 200), # Waypoint 2 (chld_N, chld_E, chld_U)
            (1000, 1000, 300),  # Waypoint 3 (wp3_N, wp3_E, wp3_U)
            (3000, 1100, 150),  # Waypoint 4 (wp4_N, wp4_E, wp4_U)
            (4000, 1200, 150),  # Waypoint 5 (wp5_N, wp5_E, wp5_U)
            (5000, 1300, 200),  # Waypoint 6 (wp6_N, wp6_E, wp6_U)
                                # Add more waypoints here as many as required
            ]


    # Initialize waypoints and initial velocities
    waypoints = [None]*3
    v_f_parent = None

    v_f_list = []
    v_0_list  = []
    # Iterate through the list of waypoints
    for i in range(len(list_waypoints) - 1):
        if i == 0:
            # On the first iteration we need to create non-existent Grandparent waypoint
            waypoints[0] = list_waypoints[i]
            waypoints[1] = list_waypoints[i]
            waypoints[2] = list_waypoints[i + 1]

        else:
            waypoints[0] = list_waypoints[i - 1]
            waypoints[1] = list_waypoints[i]
            waypoints[2] = list_waypoints[i + 1]


        # Initialize UAV flight model with the provided initial conditions
        # Iterate calculations through the waypoints
        flight_parameters = UAVFlightModel(
            m=6.0, f_t=110, csa=0.4, c_d=0.8, g=9.81, 
            v_0=0.0, rho=1.225, theta_limit=35, 
            thrust_min_percent=2,delta_time=0.1)

        # Simulate flight
        v_0, v_f, flight_time, _ = flight_parameters.simulate_flight(v_f_parent, waypoints=waypoints)
        v_f_list.append(v_f)
        v_0_list.append(v_0)
        v_f_parent = v_f

    print(f'initial velocities : {v_0_list}')
    print(f'final velocities : {v_f_list}')