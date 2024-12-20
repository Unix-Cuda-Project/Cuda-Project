#include "header.h"

__global__ void processTile(int *tile_data, int start_x, int start_y, int width,
                            int tile_size) {
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;

  int global_x = start_x + thread_x;
  int global_y = start_y + thread_y;
  int global_idx = global_y * width + global_x;

  int tile_idx = thread_y * tile_size + thread_x;

  tile_data[tile_idx] = global_idx;
}

void runProcess(int *h_tile_data, int sm_id, int tile_size) {
  int partition_size = WIDTH / tile_size;
  int block_x, block_y;
  int start_x, start_y;
  const int TILE_DATA_SIZE = partition_size * partition_size;
  char filename[100];

  int *d_tile_data;
  dim3 threadsPerBlock(partition_size, partition_size);

  cudaMalloc((void **)&d_tile_data, TILE_DATA_SIZE * sizeof(int));

  for (int i = sm_id; i < tile_size * tile_size; i += 8) {
    block_x = i % tile_size;
    block_y = i / tile_size;

    start_x = block_x * partition_size;
    start_y = block_y * partition_size;
    processTile<<<1, threadsPerBlock>>>(d_tile_data, start_x, start_y, WIDTH,
                                        partition_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_tile_data + (i / 8) * TILE_DATA_SIZE, d_tile_data,
               TILE_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    memset(filename, 0, sizeof(filename));
    createFilename(filename, "sm", "_data_", sm_id, i / 8);
    dirCat(filename, sm_id, "sm");
    writeDataToFile(filename, h_tile_data + (i / 8) * TILE_DATA_SIZE,
                    TILE_DATA_SIZE);
  }

  cudaFree(d_tile_data);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <blockIdx.x> <blockIdx.y>\n", argv[0]);
    return -1;
  }

  key_t key;
  int msgid;

  int sm_id = atoi(argv[1]);
  int tile_size = atoi(argv[2]);
  int partition_size = WIDTH / tile_size;
  int h_tile_data[partition_size * partition_size * 8] = {0};
  int final_data[WIDTH * WIDTH / 8];

  int sock = 0, sock2 = 0;
  struct sockaddr_in serv_addr;
  struct timeval start, end;
  long seconds, microseconds;
  double ms;

  key = ftok(".", 65);
  if (key == -1)
    perror("ftok");

  msgid = msgget(key, 0666 | IPC_CREAT);

  runProcess(h_tile_data, sm_id, tile_size);
  memset((int *)final_data, -1, sizeof(final_data));
  // h_tile_data에 각 sm의 데이터가 모임.
  processSM(sm_id, h_tile_data, msgid, 8, tile_size, final_data);

  gettimeofday(&start, NULL);
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0 ||
      (sock2 = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    err_exit("Socket");

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(PORT);
  serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

  if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    err_exit("Connect");

  serv_addr.sin_port = htons(PORT + 1);

  if (connect(sock2, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    err_exit("Connect");

  for (int i = 0; i < WIDTH * WIDTH / 16;
       i += partition_size * partition_size) {
    if (send(sock, (int *)(final_data + i),
             sizeof(int) * partition_size * partition_size, 0) < 0)
      err_exit("Send");
    if (send(sock2, (int *)(final_data + WIDTH * WIDTH / 16 + i),
             sizeof(int) * partition_size * partition_size, 0) < 0)
      err_exit("Send2");
  }
  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  ms = (seconds * 1000) + (microseconds / 1000.0);

  recv(sock, final_data, 1, 0);
  recv(sock2, final_data, 1, 0);

  close(sock);
  close(sock2);

  printf("sm%d Client-Server Communication : %f ms\n", sm_id, ms);

  return 0;
}
