# RPL SQuEAK Filter

The **S**imple **QU**aternion **E**xtended **A**ttitude **K**alman Filter.

That's *attitude*, not altitude.

## Compiling and running

To run the simulation tests:

1. Install Eigen3

2. Run

```bash
mkdir build && cd build && mkdir output
cmake ..        # generate Makefile
cmake --build . # build executable
./run           # run tests, output written to ./output
```
