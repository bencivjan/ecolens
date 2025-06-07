#!/bin/bash

# for i in $(seq 1 3); do

python3 simulation.py ecolens \
    --frame-dir ../filter-images/ground-truth-JH-full \
    --energy-profile ../viz/energy-JH-1.csv \
    --accuracy-profile ../viz/accuracy-JH-2-dynamic.csv \
    --log-file ecolens-JH-day-0.9.csv \
    --target-accuracy 0.90 \
    --explore-time 5 \
    --exploit-time 60

python3 simulation.py ecolens \
    --frame-dir ../filter-images/ground-truth-JH-night-full \
    --energy-profile ../viz/energy-JH-night-1.csv \
    --accuracy-profile ../viz/accuracy-JH-night-1.csv \
    --log-file ecolens-JH-night-0.9.csv \
    --target-accuracy 0.90 \
    --explore-time 5 \
    --exploit-time 60

python3 simulation.py ecolens \
    --frame-dir ../filter-images/ground-truth-Alma-full \
    --energy-profile ../viz/energy-Alma.csv \
    --accuracy-profile ../viz/accuracy-Alma.csv \
    --log-file ecolens-Alma-0.9.csv \
    --target-accuracy 0.90 \
    --explore-time 5 \
    --exploit-time 60

# done