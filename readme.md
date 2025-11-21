# Accelerated Learning for Neural Model Predictive Control

<video width="640" height="480" autoplay>
    <source src="combined_output.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

This is the source code of paper:

    @article{li2025TrainingNMPC,
        title={Accelerated Learning for Neural Model Predictive Control},
        author={Anran Li, John P. Swensen, and Mehdi Hosseinzadeh},
        journal={International Journal of Robust and Nonlinear Control},
        year={2025},
        note={Currently under review}
    }

The code provides both simulation and real-time control source code. 

If you find it is helpful, please cite our paper.

The simulation 

$$
\begin{align}
x(t+1) &= \begin{bmatrix}
0 & -0.0005 \\
1 & -0.0965
\end{bmatrix} x(t) +
\begin{bmatrix}
0.0004 \\
-0.00
\end{bmatrix} u(t), \\
y(t) &= \begin{bmatrix}
0 & 1
\end{bmatrix} x(t) + T_{amb.},
\end{align}
$$


where $T_{amb.}$ is ambient temperature.

If you want to compare your results with our heater system, this is a good demo.
## 
    heater_new_model_simulation

* Dynamic simulation with LQR control, Multi-Agent optimal control, and Neural Controller methods.

## Hardware Experiemnt:
## 

    heater_new_model_hardware.m

* The second section drow the paper figures:
(Need to load both .mat files.)
![Alt text](heater_res.png )

* The first second connects Arduino kit board, build transition with the heater system.


## Other Files
    heater_new_model_multi_agent_datacollection.m

* Used for data collection. 
* A little different from the method metioned in paper. This reference code shows data is collected every iterations given one reference $T_{desire}$.
##
    NN_heater_trajectories_simulation.py

* Verify the convergence for different trained NN models.
