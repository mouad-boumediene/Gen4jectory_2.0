# Gen4jectory Algorithm – 4-D Trajectory Planning for Rotary-Wing UAVs

## Overview
Gen4jectory 2.0 is a 4-D trajectory-planning framework for fleets of rotary-wing UAVs. It employs a rigorous OBB-vs-OBB collision check based on the Separating Axis Theorem (SAT) to guarantee zero Loss-of-Separation (LoS) events within each drone's separation volume. By fusing high-fidelity rotorcraft performance models with a traffic-aware pathfinder and time-parameterised trajectory smoothing, Gen4jectory 2.0 rapidly generates conflict-free schedules that honour individual aircraft kinematics and air-space constraints. The project includes detailed physical modelling, map representation, trajectory generation, and outlines future research directions.
## Project Objectives
Developing a multi-drone trajectory planning algorithm for air-space safety.

## Team Members
| Ivan Panov | Mouad Boumediene |
|:--------------------------------:|:----------------------------------------:|
| [![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/SkyIvanCoding) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ivan-panov-0ba21476/) | [![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/mouad-boumediene) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mouad-boumediene/) [![YouTube](https://img.shields.io/badge/-YouTube-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/channel/UCxeDM47jeD0CQTCTHJPzZaw) [![Website](https://img.shields.io/badge/-Website-000000?style=flat-square&logo=web&logoColor=white)](https://mouadboumediene.com) |

## Requirements
This project has been created in Python 3.12.5 using the following:

- colorama==0.4.4
- matplotlib==3.5.2
- networkx==2.8.8
- numpy==1.21.2
- pandas==1.4.1
- raylib==5.0.0.2
- scipy==1.8.0
- tqdm==4.64.0

## How to Run
- Run `main.py` for a single visualized simulation.
- Run `multiple_experiments_parallel.py` for performing a batch of experiments.
  
## Hardware and software
This project was developed on a desktop PC equipped with a 13th Gen Intel i9 16-core CPU and 32 GB of RAM, running Windows 11.
All simulations were implemented in Python 3.12.5
