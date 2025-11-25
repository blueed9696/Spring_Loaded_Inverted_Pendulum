# Spring_Loaded_Inverted_Pendulum

This project simulates a Spring-Loaded Inverted Pendulum (SLIP) model, demonstrating the dynamics of a hopping robot or animal.

## Simulation

<img src="assets/spring.gif" width="800">

## Project Structure

- **`model.py`**: Defines the physical parameters of the system (mass, spring constant, gravity, original spring length) and the equations of motion for both flight and stance phases.
- **`simulation.py`**: It handles the numerical integration of the equations of motion by solving non-linear equations. It manages the transition between flight and stance phases.
- **`state.py`**: Contains data classes (`SLIPState`, `ApexState`, `TouchDownState`) and enums (`Phase`) to represent the state of the robot at different points in the gait cycle.
- **`assets/`**: Directory containing photo and video.

