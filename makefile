CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
CXX       = nvcc
CXXFLAGS  = -std=c++17 -O2 -I"$(CUDA_PATH)\include"

mosaic.exe: src/main.cu
	$(CXX) $(CXXFLAGS) -o mosaic.exe src/main.cu