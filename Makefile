# Makefile

# 컴파일러 설정
NVCC = nvcc
CC = gcc

# 컴파일 플래그 설정
NVCC_FLAGS = -arch=sm_75 -I.
CFLAGS = -Wall

# 타겟 실행 파일 이름
CLIENT_NAME = client
SERVER_NAME = server

SRC_DIR = src
OBJ_DIR = obj

SM_DIR =	sm0 \
					sm1 \
					sm2 \
					sm3 \
					sm4 \
					sm5 \
					sm6 \
					sm7 

SERV_DIR	=	server0 \
						server1

# 소스 파일
CUDA_SRC =	$(SRC_DIR)/smCuda.cu \
						$(SRC_DIR)/file_create.c \
						$(SRC_DIR)/data_exchange.c
C_SRC = 		$(SRC_DIR)/main.c 
SERVER_SRC = $(SRC_DIR)/server.c

# 오브젝트 파일
CUDA_OBJ =	$(OBJ_DIR)/smCuda.o \
						$(OBJ_DIR)/file_create.o \
						$(OBJ_DIR)/data_exchange.o
C_OBJ =			$(OBJ_DIR)/main.o
SERVER_OBJ =	$(OBJ_DIR)/server.o \
							$(OBJ_DIR)/file_create.o

# 기본 타겟
all: $(CLIENT_NAME) $(SERVER_NAME)
# $(CC) $(CFLAGS) $(CUDA_OBJ) -o a.out -lcudart -L/usr/lib/x86_64-linux-gnu

$(CLIENT_NAME): $(CUDA_OBJ) $(C_OBJ)
	@mkdir -p $(SM_DIR)
	$(CC) $(CFLAGS) $(C_OBJ) -o $@
	$(NVCC) $(NVCC_FLAGS) -o a.out $(CUDA_OBJ)

$(SERVER_NAME): $(SERVER_OBJ)
	@mkdir -p $(SERV_DIR)
	$(CC) $(CFLAGS) $(SERVER_OBJ) -o $@ 

# CUDA 소스 파일 컴파일
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# C 소스 파일 컴파일
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# 실행 파일 링크

# 클린업
clean:
	rm -rf $(SERVER_OBJ) $(CUDA_OBJ) $(C_OBJ)
	find . -name "*.txt" -type f -delete

fclean:
	make clean
	rm -rf $(CLIENT_NAME) $(SERVER_NAME) $(SERV_DIR) $(SM_DIR)  a.out

re:
	make fclean
	make all

.PHONY: all clean re fclean
