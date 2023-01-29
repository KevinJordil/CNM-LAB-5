OBJS	= main.o
SOURCE	= main.cu
HEADER	= 
OUT	= main
CC	 = nvcc
FLAGS	 = -O3 -c -I /usr/include/opencv4
LFLAGS	 = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

#  -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
all: $(OUT)

$(OUT): $(OBJS)
	$(CC) $(OBJS) -o $(OUT) $(LFLAGS)

main.o: $(SOURCE)
	$(CC) $(FLAGS) -o $@ $<

clean:
	rm -f $(OBJS) $(OUT)