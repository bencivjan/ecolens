# EcoLens: Leveraging Multi-Objective Bayesian Optimization for Energy-Efficient Video Processing on Edge Devices

The EcoLens system reduces energy consumption on an energy-limited streaming edge device (e.g. camera) while maintaining a target downstream object-detection accuracy.

![EcoLens Diagram](viz/figures/ecolens-design-diagram(3).png)

The system is composed of an offline stage and online stage.

## Offline
- Offline Camera: `raspberrypi/profiler.py`
- Offline Server:
    - Accuracy Evaluator: `evaluate.py`

## Online
- Online Camera: `raspberrypi/stream_simulator.py`
- Online Server:
    - MBO Engine: `simulation/bayesian_opt.py`
    - Configuration Evaluation: `simulation/evaluator.py`
    - Algorithm Simulation: `simulation/simulation.py`

