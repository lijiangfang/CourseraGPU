# IDIR=./
CXX = nvcc

all: clean build

#build: bias_relu_kernel.cu imgPP.cpp lenet5.cpp ocr.cpp
#	$(CXX) bias_relu_kernel.cu imgPP.cpp lenet5.cpp ocr.cpp -o ocr.exe -std=c++17 \
#	-I/usr/local/cuda/include \
#	-I/usr/local/cuda/targets/x86_64-linux/include \
#	-L/usr/local/cuda/lib64 \
#	-lcublas -lcudnn -lcudart -lcuda -lnppicc -lnppig -lnppial -lnppidei -lnppif -lnppist -lnppisu -lnppc

build: ocr.exe
ocr.exe: bias_relu_kernel.o imgPP.o lenet5.o ocr.o
	$(CXX) bias_relu_kernel.o imgPP.o lenet5.o ocr.o -o ocr.exe -std=c++17 \
	-L/usr/local/cuda/lib64 \
	-lcublas -lcudnn -lcudart -lcuda -lnppicc -lnppig -lnppial -lnppidei -lnppif -lnppist -lnppisu -lnppc

bias_relu_kernel.o: bias_relu_kernel.cu bias_relu_kernel.h
	$(CXX) -c bias_relu_kernel.cu -o bias_relu_kernel.o -std=c++17 \
	-I/usr/local/cuda/include \
	-I/usr/local/cuda/targets/x86_64-linux/include 

imgPP.o: imgPP.cpp cucheck.h imgPP.h
	$(CXX) -c imgPP.cpp -o imgPP.o -std=c++17 \
	-I/usr/local/cuda/include \
	-I/usr/local/cuda/targets/x86_64-linux/include 

lenet5.o: lenet5.cpp lenet5.h cucheck.h bias_relu_kernel.h
	$(CXX) -c lenet5.cpp -o lenet5.o -std=c++17 \
	-I/usr/local/cuda/include \
	-I/usr/local/cuda/targets/x86_64-linux/include 

ocr.o: ocr.cpp imgPP.h lenet5.h
	$(CXX) -c ocr.cpp -o ocr.o -std=c++17 \
	-I/usr/local/cuda/include \
	-I/usr/local/cuda/targets/x86_64-linux/include 

run:
	./ocr.exe $(ARGS)

clean:
	rm -f ocr.exe output*.txt *.o
