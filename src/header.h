#pragma once

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define WIDTH 16
#define PORT 8080
#define MAX_CLIENTS 8
#define BUFFER_SIZE 1024

struct mymsgbuf {
  long mtype;
  int mtext[1024];
};

#ifdef __cplusplus
extern "C" {
#endif

void writeDataToFile(const char *filename, int *data,
                     int data_size);
void createFilename(char *filename, const char *str1,
                    const char *str2, int id, int i);

void runProcess(int *h_tile_data, int sm_id, int tile_size);
void processSM(int sm_id, int *sm_data, int msg_queue_id,
               int num_procs, int tile_size,
               int *final_data);

void dirCat(char *filename, int id, const char *str);
void err_exit(const char *str);

#ifdef __cplusplus
}
#endif
