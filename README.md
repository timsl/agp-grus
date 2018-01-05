# agp-grus

## Team Grus - DD2360
Konrad Magnusson

Tim Olsson

Alex Sundström

Victor Ähdel

## Running
You will need `glm`, `glfw`, `OpenGL`, and `CUDA` installed somewhere in PATH.

Then to run it, do:
```sh
cd source
make run
```

Use the mouse (or arrow keys) to look around, and W/A/S/D/CTRL/SHIFT
to pan around. Further, press dot for a single update step or P to
start/stop the simulation. R resets the camera if you end up lost
(though the planets may have moved away).

## Command line arguments
`make run` will run with a default number of particles and block size (number of kernels that share memory and synchronize in the CUDA part), but those parameters can otherwise be specified from the command line.
`./bin/moon 1000 128` would instead run 1000 particles with a block size of 128.

For scripting, a third parameter can be specified which then makes the program run that many iterations before quitting, and not syncing to the monitors refresh rate (vsync), to get better timings.

A csv file called simply `log` is created in either case, which contains timings for different parts from each update iteration.

