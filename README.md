# CDI learning

## Description
This repository was created to study how CDI (Coherent Diffraction Imaging) algorithms work. The project help perform the first steps in studying lensless X-ray microscopy.

## How to Run
1. Clone the repository or download the script:
   ```bash
   git clone https://github.com/dorka63/cdi-learning.git
   ```
## Branching Strategy
- `main`: The stable branch. All production-ready code and required files are merged here.
- `task1-cpp`: C++ implementation of the phase-retrieval algorithm. Requires the OpenCV package.
- `task2`: The branch used for embedding the virtual-lens method. Features and fixes are merged here before being released to `main`.

## Code Versions
This repository contains three versions of the code for phase retrieval in the far zone:
- **GPU Version**: `task1_GPU.py` — Implements the phase-retrieval algorithm using GPU acceleration. Requires a GPU and the `cupy` library.
- **CPU Version**: `task1.py` — Implements the phase-retrieval algorithm using CPU. Requires `numpy` and other standard libraries.
- **C++ Version**: `test-opencv.cpp` — Implements the phase-retrieval algorithm in C++. Requires the OpenCV package and uses the R50 random number generator (forked from [Parallel-RNG-using-GPU](https://github.com/AdroitAnandAI/Parallel-RNG-using-GPU)).
  
## Acknowledgements
This project is based on the following research papers:
1. Fienup, James R. "Phase retrieval algorithms: a comparison." Applied optics 21.15 (1982). [DOI: 10.1364/AO.21.002758](https://doi.org/10.1364/AO.21.002758)  
   - This paper provides the theoretical foundation for the phase-retrieval algorithms implemented in this project.
2. Artyukov, I.A., Vinogradov, A.V., Gorbunkov, M.V. et al. "Virtual Lens Method for Near-Field Phase Retrieval." Bull. Lebedev Phys. Inst. 50, 414–419 (2023). [DOI: 10.3103/S1068335623100020](https://doi.org/10.3103/S1068335623100020)  
   - This paper introduces the virtual-lens method and relaxation strategy of HIO/ER aglgorithms usage.
