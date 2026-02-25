Since we implemented the get_docs function in C++, a pdf parser tool called poppler nedds to be installed. The steps are:

module load cmake gcc

mkdir -p $HOME/local/poppler

cd $HOME
wget https://poppler.freedesktop.org/poppler-24.03.0.tar.xz
tar -xf poppler-24.03.0.tar.xz
cd poppler-24.03.0

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local/poppler -DENABLE_UNSTABLE_API_ABI_HEADERS=ON -DENABLE_CPP=ON
make -j
make install

export LD_LIBRARY_PATH=$HOME/local/poppler/lib:$LD_LIBRARY_PATH (for good measures)

And then compile with:

g++ get_docs.cpp -fopenmp -I$HOME/local/poppler/include/poppler -L$HOME/local/poppler/lib -lpoppler-cpp -o get_docs


All of these steps are necessary because we do not have sudo access


****


If you wish to try out the pdf download C++ implementation, please use mpicc and mpirun as it initially distributes xml chunk across MPI processes


The main_program.py does everythong automatically and outputs the runtimes in the console. We manually noted the runtimes and made the charts.

Dont forget to set the number of threads with:
export OMP_NUM_THREADS=x

The other main dependency required to run this is the vllm library. It takes a while and may not fit in the allocated disk quota.
I used scratch space to install vllm and configured the python environment so that everything points to the specific scratch disk space.

Depending on your path to your specific scratch location, might need to change the import statement for vllm (to point to the correct path)


In summary the main dependencies are as follows:

1. Poppler - C++ pdf parsing library
2. vllm
3. tinyxml2 - For xml procssing in C++ (downloading pdf stage). This is already present among the code files

****

After everything as been setup, just run the "main_program.py" with python and the console should print out everything we used for creating the plots and tables.
