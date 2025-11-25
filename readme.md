# Accelerated Learning for Neural Model Predictive Control

Drone hovering results from our trained neural Model Predictive Control(MPC):
<p align="center">
  <img src="combined_high_quality.gif" width="800" alt="demo">
</p>


The neural MPC is trained following the novel accelerated training idea in our paper:

    @article{li2025TrainingNMPC,
        title={Accelerated Learning for Neural Model Predictive Control},
        author={Anran Li, John P. Swensen, and Mehdi Hosseinzadeh},
        journal={International Journal of Robust and Nonlinear Control},
        year={2025},
        note={Under review}
    }

The repo opens all codes for supporting the above paper, including:

1. State-Space Models:
    - Parrot Bebop 2 Drone
    - TCLab Thermal Kit
2. Simulations of the above models
3. Proposed accelerated data collection and training/testing solutions 

If you find it is helpful, please cite our paper.

## 1. Parrot Bebop 2 Drone Hovering Model (State-Space)

$$
\begin{align}
x(t+1) &=
\begin{bmatrix}
    0 & 1 \\
    0 & -1.7873
\end{bmatrix} x(t)
+
\begin{bmatrix}
    0 \\
    -1.7382
\end{bmatrix} u(t)\nonumber,\\
y(t) &= \begin{bmatrix}
1 & 0
\end{bmatrix} x(t) \nonumber
\end{align}
$$

Above model is applied to Simulation and Real-Time control experiments.

### Drone Model Simulation:
See ``semi_supervison`` folder:
- Neural MPC training: 
``    semi_supervison/train_DroneZ_MPC_multi_xr.py``

- MPC-inspired Loss: ``semi_supervison/Objective_Formulations_mpc.py``

- Sampling Methods:
``semi_supervison/initial_sampling_pts_gen.py``

- Sythesis Data Updates:
``semi_supervison/UpdatingDataset.py``

Drone Simulation:

- MATLAB:
``DroneZ_MPC/Real_Drone_Experiments/drone_simulation_random_point.m``

- Python:
``semi_supervison/initial_sampling_pts_gen.py  (end of comment sections)``

### Drone Model Real-Time Control:
See folder ``DroneZ_MPC``:

- Testing Dataset Collection:
``DroneZ_MPC/droneZ_MPC_datacollection.m``

- Connect and Control Bebop 2:
``DroneZ_MPC/Real_Drone_Experiments/Continue_LQR_NN_Z_Controller.m``

The neural MPC weights are trained follows the code from simulation section.

## 2. TCLab Thermal Model

$$
\begin{align}
x(t+1) &= \begin{bmatrix}
0 & -0.0005 \\
1 & -0.0965
\end{bmatrix} x(t) +
\begin{bmatrix}
0.0004 \\
-0.00
\end{bmatrix} u(t),\nonumber \\
y(t) &= \begin{bmatrix}
0 & 1
\end{bmatrix} x(t) + T_{amb.}\nonumber,
\end{align}
$$


where $T_{amb.}$ is ambient temperature.

Code is located on the main path.

### Simulation and Real-Time Control:

- Neural MPC Training:
``train_Heater_multi_xr.py``

- Synthesis Dataset:
``Heater_Dataset.py``

Simulation code:
- MATLAB:
``heater_new_model_simulation_varyXr.m``

- Python:
``NN_heater_trajectories_simulation.py``

Change control input to your control algorithms accordingly.

Real-Time Thermal System Control:
``heater_new_model_hardware_varyXr.m``


