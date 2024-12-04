# Makefile

# 컴파일러 설정
NVCC = nvcc
CC = gcc

# 컴파일 플래그 설정
NVCC_FLAGS = -arch=sm_75
CFLAGS = -Wall

# 타겟 실행 파일 이름
NAME = test

# 소스 파일
CUDA_SRC = smCuda.cu
C_SRC = main.c

# 오브젝트 파일
CUDA_OBJ = smCuda.o
C_OBJ = main.o

# 기본 타겟
all: $(NAME)

# CUDA 소스 파일 컴파일
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) $< -g

# C 소스 파일 컴파일
$(C_OBJ): $(C_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

# 실행 파일 링크
$(NAME): $(CUDA_OBJ) $(C_OBJ)
	$(CC) $(CFLAGS) $(C_OBJ) -o $@

# 클린업
clean:
	rm -f $(NAME) $(CUDA_OBJ) $(C_OBJ) *.txt a.out

.PHONY: all clean