# Readme for CFD Master Praktikum of Group G

#### Author: Yue Zhu, Yingqiang Gao and Nikhil Agarwal

Our code is able to not only simulate the 5 problems in the worksheet2 with a short command, but also available to deal with other cases with or without heat while being provided enough information with input data of variables and the geometry. 

To run it, you start a terminal and switch to where the main.c is. You need to build it first by typing "make" in the terminal. Then if everything is right, you can find the executable "sim" being created and then you can run with the command:

"./sim karman" for the karman vortex street problem;
"./sim step" for the fluid over a trap problem;
"./sim natural1" for the natural convection problem with the 1st group of parameters;
"./sim natural2" for the natural convection problem with the 2nd group of parameters;
"./sim trap" for the heat trap problem;
"./sim trapinverse" for the heat trap problem if you inverse the hot wall and the cold wall;
"./sim rbc" for the rayleigh-benard convection problem

General cases:

"./sim (problem) (geometry) (Output_filename) fluid" for a general fluid case without heat.

Example: "./sim karman.dat karman.pgm output/karman fluid"

"./sim (problem) (geometry) (Output_filename) heat downup/leftright (Th) (Tc)" for a general fluid case with heat. downup/leftright are where the wall are Th and Tc.

Example: "./sim natural1.dat natural1.pgm output/natural1 heat leftright 1.0 0.0"

## Problem 1.4 a) The Karman Vortex Street:

The flow in a channel hits a tilted plate. At the left boundary, the fluid inflow
has a constant velocity profile (u = 1:0, v = 0:0), while at the upper and lower
boundaries no-slip conditions are imposed. The plate occupies one fifth of the
channel width and is three cells thick. The distance from the left boundary is
assumed to be the same as the distance to the lower and upper boundaries. Make
sure that there are no forbidden obstacle cells.

|             |            |              |                |
| ----------- | :--------: | -----------: | -------------: |
| imax = 100  | jmax = 20  | xlength = 10 | ylength = 2    |
| dt = 0.05   | t_end = 20 | tau = 0.5    | dt_value = 2.0 |
| eps = 0.001 | omg = 1.7  | alpha = 0.9  | itermax = 500  |
| GX = 0.0    | GY = 0.0   | Re = 10000   |                |
| UI = 1.0    | VI = 0.0   | PI = 0.0     |                |
|             |            |              |                |

### Geometry
![Karman Geometry](./Images/karman_geometry.png)

### Velocity Streamlines
![Karman Velocity Stream Lines](./Images/karman_stream.png)

The streamlines represent the flow of fluid particles in the domain.
**Observation:**  The vortices are formed from alternating sides of the object after hitting the obstacle in the middle.

### Velocity Vector Glyph
![Karman Velocity Glyph](./Images/karman_velocity.png)

**Observation:** Velocity Vector Glyph represents the fluid velocity vector at any cell.

### Pressure
![Karman Pressure](./Images/karman_pressure.png)
**Observation:** The pressure remains constant in the entire domain because the fluid is incompressible and we neglect the effect of temperature.

## Problem 1.4 b) Flow over a Step (When UI = 0.5):

The fluid flows through a channel widening on one side. No-slip conditions are
imposed at the upper and lower walls. The obstacle domain is represented by
a square filling up half of the channel height.

|             |             |              |                 |
| ----------- | :---------: | -----------: | --------------: |
| imax = 100  | jmax = 20   | xlength = 10 | ylength = 2     |
| dt = 0.05   | t_end = 500 | tau = 0.5    | dt_value = 10.0 |
| eps = 0.001 | omg = 1.7   | alpha = 0.9  | itermax = 500   |
| GX = 0.0    | GY = 0.0    | Re = 100     |                 |
| UI = 0.5    | VI = 0.0    | PI = 0.0     |                 |
|             |             |              |                 |

### Geometry
![Step Geometry](./Images/step_geometry.png)

### Velocity Streamlines
![Step Velocity Stream Lines](./Images/step_stream.png)

### Velocity Vector Glyph
![Step Velocity Glyph](./Images/step_velocity.png)

### Pressure
![Step Pressure](./Images/step_pressure.png)

## Problem 1.4 b) Flow over a Step (When UI = 0.0):

The fluid flows through a channel widening on one side. No-slip conditions are
imposed at the upper and lower walls. The obstacle domain is represented by
a square filling up half of the channel height.

|             |             |              |                 |
| ----------- | :---------: | -----------: | --------------: |
| imax = 100  | jmax = 20   | xlength = 10 | ylength = 2     |
| dt = 0.05   | t_end = 500 | tau = 0.5    | dt_value = 10.0 |
| eps = 0.001 | omg = 1.7   | alpha = 0.9  | itermax = 500   |
| GX = 0.0    | GY = 0.0    | Re = 100     |                 |
| UI = 0.0    | VI = 0.0    | PI = 0.0     |                 |
|             |             |              |                 |

### Velocity Streamlines
![Step Velocity Stream Lines](./Images/step_stream_orig.jpeg)

### Velocity Vector Glyph
![Step Velocity Glyph](./Images/step_velocity_orig.jpeg)

### Pressure
![Step Pressure](./Images/step_pressure_orig.jpeg)

## Problem 2.6 c) Natural Convection with 1st set of Parameters:

All boundaries have no-slip conditions.

|               |              |             |                 |
| ------------- | :----------: | ----------: | --------------: |
| imax = 50     | jmax = 50    | xlength = 1 | ylength = 1     |
| dt = 0.05     | t_end = 1000 | tau = 0.5   | dt_value = 10.0 |
| eps = 0.00001 | omg = 1.7    | alpha = 0.5 | itermax = 100   |
| GX = 0.0      | GY = -1.1    | Re = 1000   | PR = 7          |
| UI = 0.0      | VI = 0.0     | PI = 0.0    |                 |
| TI = 0.0      | T_h = 1.0    | T_c = 0.0   | beta = 0.00021  |
|               |              |             |                 |

### Geometry
![Natural1 Geometry](./Images/natural1_geometry.png)

### Velocity Streamlines
![Natural1 Velocity Stream Lines](./Images/natural1_velocity_stream.png)

### Velocity Vector Glyph
![Natural1 Velocity Glyph](./Images/natural1_velocity_glyph.png)

### Pressure
![Natural1 Pressure](./Images/natural1_pressure.png)

### Temperature
![Natural1 Pressure](./Images/natural1_temperature.png)

## Problem 2.6 c) Natural Convection with 2nd set of Parameters:
All boundaries have no-slip conditions.

|               |               |             |                 |
| ------------- | :-----------: | ----------: | --------------: |
| imax = 50     | jmax = 50     | xlength = 1 | ylength = 1     |
| dt = 0.05     | t_end = 10000 | tau = 0.5   | dt_value = 50.0 |
| eps = 0.00001 | omg = 1.7     | alpha = 0.5 | itermax = 100   |
| GX = 0.0      | GY = -1.1     | Re = 20000  | PR = 7          |
| UI = 0.0      | VI = 0.0      | PI = 0.0    |                 |
| TI = 0.0      | T_h = 1.0     | T_c = 0.0   | beta = 0.00021  |
|               |               |             |                 |

### Geometry
![Natural2 Geometry](./Images/natural1_geometry.png)

### Velocity Streamlines
![Natural2 Velocity Stream Lines](./Images/natural2_velocity_stream.png)

### Velocity Vector Glyph
![Natural2 Velocity Glyph](./Images/natural2_velocity_glyph.png)

### Pressure
![Natural2 Pressure](./Images/natural2_pressure.png)

### Temperature
![Natural2 Pressure](./Images/natural2_temperature.png)

## Problem 2.6 d) Fluid Trap (when the left wall is hotter than the right wall):

|               |              |             |                 |
| ------------- | :----------: | ----------: | --------------: |
| imax = 100    | jmax = 50    | xlength = 2 | ylength = 1     |
| dt = 0.05     | t_end = 2000 | tau = 0.5   | dt_value = 10.0 |
| eps = 0.00001 | omg = 1.7    | alpha = 0.5 | itermax = 1000  |
| GX = 0.0      | GY = -9.81   | Re = 10000  | PR = 7          |
| UI = 0.0      | VI = 0.0     | PI = 0.0    |                 |
| TI = 0.0      | T_h = 0.5    | T_c = -0.5  | beta = 0.00063  |
|               |              |             |                 |

### Geometry
![Trap Geometry](./Images/trap_geometry.png)
All boundaries have no-slip conditions.

### Velocity Streamlines
![Trap Velocity Stream Lines](./Images/trap_velocity_stream.png)

### Velocity Vector Glyph
![Trap Velocity Glyph](./Images/trap_velocity_glyph.png)

### Pressure
![Trap Pressure](./Images/trap_pressure.png)

### Temperature
![Trap Pressure](./Images/trap_temperature.png)

## Problem 2.6 d) Fluid Trap (when the left wall is colder than the right wall):

All boundaries have no-slip conditions.

|               |              |             |                 |
| ------------- | :----------: | ----------: | --------------: |
| imax = 100    | jmax = 50    | xlength = 2 | ylength = 1     |
| dt = 0.05     | t_end = 2000 | tau = 0.5   | dt_value = 10.0 |
| eps = 0.00001 | omg = 1.7    | alpha = 0.5 | itermax = 1000  |
| GX = 0.0      | GY = -9.81   | Re = 10000  | PR = 7          |
| UI = 0.0      | VI = 0.0     | PI = 0.0    |                 |
| TI = 0.0      | T_h = -0.5    | T_c = 0.5  | beta = 0.00063  |
|               |              |             |                 |

### Velocity Streamlines
![Trap Velocity Stream Lines](./Images/trap_velocity_stream_lc.png)

### Velocity Vector Glyph
![Trap Velocity Glyph](./Images/trap_velocity_glyph_lc.png)

### Pressure
![Trap Pressure](./Images/trap_pressure_lc.png)

### Temperature
![Trap Pressure](./Images/trap_temperature_lc.png)


## Problem 2.6 e) Rayleigh-B´enard Convection (Original Parameters with Geometrical aspect ratio almost equal to 5):

**Phenomenon**: Rayleigh–Bénard convection is a type of natural convection, occurring in a plane horizontal layer of fluid heated from below, in which the fluid develops a regular pattern of convection cells known as Bénard cells. Buoyancy, and hence gravity, are responsible for the appearance of convection cells.

The height of the layer is small compared to the horizontal dimension. At first, the temperature of the bottom plane is the same as the top plane. The liquid will then tend towards an equilibrium, where its temperature is the same as its surroundings. (Once there, the liquid is perfectly uniform: to an observer it would appear the same from any position. This equilibrium is also asymptotically stable: after a local, temporary perturbation of the outside temperature, it will go back to its uniform state, in line with the second law of thermodynamics).

Then, the temperature of the bottom plane is increased slightly yielding a flow of thermal energy conducted through the liquid. The system will begin to have a structure of thermal conductivity: the temperature, and the density and pressure with it, will vary linearly between the bottom and top plane. A uniform linear gradient of temperature will be established. (This system may be modelled by statistical mechanics).

Once conduction is established, the microscopic random movement spontaneously becomes ordered on a macroscopic level, forming Benard convection cells, with a characteristic correlation length.

|               |               |               |                  |
| ------------- | :-----------: | ------------: | ---------------: |
| imax = 85     | jmax = 18     | xlength = 8.5 | ylength = 1      |
| dt = 0.05     | t_end = 45000 | tau = 0.5     | dt_value = 100.0 |
| eps = 0.00001 | omg = 1.7     | alpha = 0.5   | itermax = 100    |
| GX = 0.0      | GY = -0.3924  | Re = 33.73    | PR = 12500       |
| UI = 0.0      | VI = 0.0      | PI = 0.0      |                  |
| TI = 293.0    | T_h = 294.78  | T_c = 291.20  | beta = 0.00179   |
|               |               |               |                  |

### Geometry
![Trap Geometry](./Images/rbc_geometry.png)

### Velocity Streamlines
![Trap Velocity Stream Lines](./Images/rbc_velocity_stream.png)

### Velocity Vector Glyph
![Trap Velocity Glyph](./Images/rbc_velocity_glyph.png)

### Pressure
![Trap Pressure](./Images/rbc_pressure.png)

### Temperature
![Trap Pressure](./Images/rbc_temperature.png)

___

## Problem 2.6 e) Rayleigh-B´enard Convection (Original Parameter with Geometrical aspect ratio equal to 4):

### Velocity Streamlines
![Trap Velocity Stream Lines](./Images/rbc_velocity_stream_4.png)

### Velocity Vector Glyph
![Trap Velocity Glyph](./Images/rbc_velocity_glyph_4.png)

### Pressure
![Trap Pressure](./Images/rbc_pressure_4.png)

### Temperature
![Trap Pressure](./Images/rbc_temperature_4.png)

## Problem 2.6 e) Rayleigh-B´enard Convection (Original Parameter with Geometrical aspect ratio equal to 6):

### Velocity Streamlines
![Trap Velocity Stream Lines](./Images/rbc_velocity_stream_6.png)

### Velocity Vector Glyph
![Trap Velocity Glyph](./Images/rbc_velocity_glyph_6.png)

### Pressure
![Trap Pressure](./Images/rbc_pressure_6.png)

### Temperature
![Trap Pressure](./Images/rbc_temperature_6.png)