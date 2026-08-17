# Environmental Volumetric Neural Shading of Clouds for Real-Time Rendering

This is the source code for the paper [_Environmental Volumetric Neural Shading of Clouds for Real-Time Rendering_](https://fileadmin.cs.lth.se/graphics/research/papers/2026/environmental_volumetric_neural_shading_of_clouds_for_real_time_rendering/).

## Build

Clone the repo, and make sure to also initialize the submodules:

	git clone --recurse-submodules https://github.com/LUGGPublic/evns-clouds-rtr.git

If you cloned the repo but forgot to initialize the submodules, use:

	git submodule update --init --recursive

This project uses CMake to generate project files, and has been tested on Windows with Visual Studio and Linux with Make.
Use Visual Studio's internal CMake support for setting up the project: after cloning the git repo as instructed above, start Visual studio and pick `Open a local folder` and pick the folder where the repo was cloned to.

On other platforms, use the CMake command line interface to setup the project.
Navigate to the folder where the repo resides:

	cd evns-clouds-rtr

Create a build folder:

	mkdir build

Change into the build folder and initialize CMake:

	cd build
	cmake ..

To build and compile the project use:

	cmake --build .

To run the project, change to the binary output folder and run the executable file:

	cd {Debug,Release}/bin
	./Clouds.bin

## Resources

One dataset will be downloaded as part of the CMake initialization from Lund University's server [Link to resources (136 MB)](https://fileadmin.cs.lth.se/graphics/research/papers/2026/environmental_volumetric_neural_shading_of_clouds_for_real_time_rendering/evns_clouds_rtr_resources.zip).
An example of how to setup Blender, as described in section 3.2 in the paper, is provided here [Link to Blender file using Walt Disney Animation Studio Cloud (315 MB, CC 3.0 BY-SA)](https://fileadmin.cs.lth.se/graphics/research/papers/2026/environmental_volumetric_neural_shading_of_clouds_for_real_time_rendering/wdas_mesh.zip)
