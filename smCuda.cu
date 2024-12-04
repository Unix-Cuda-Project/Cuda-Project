#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <unistd.h>

#define WIDTH 16 // 전체 데이터의 가로 길이

struct mymsgbuf {
  long mtype;
  int mtext[1024];
};

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

// 데이터 분할 프로세스.
void runProcess(int *h_tile_data, int sm_id, int tile_size) {
  int partition_size = WIDTH / tile_size;
  int block_x, block_y;
  int start_x, start_y;
  const int TILE_DATA_SIZE = partition_size * partition_size;

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
  }

  cudaFree(d_tile_data);
}

//생성된 데이터를 메시지 큐로 서로 간에 데이터 공유.
void processSM(int sm_id, int *sm_data, int msg_queue_id, int num_procs,
               int tile_size) {
  struct mymsgbuf msg;
  int total_data_size = WIDTH * WIDTH;
  int final_data[total_data_size / num_procs];

  int msg_size = WIDTH / tile_size;
  for (int i = 0; i < total_data_size / num_procs; i += msg_size) {
    msg.mtype = sm_data[i] + 1;
    for (int j = 0; j < msg_size; j++) {
      msg.mtext[j] = sm_data[i + j];
    }
    if (msgsnd(msg_queue_id, &msg, sizeof(int) * msg_size, 0) == -1) {
      perror("msgsnd");
      exit(1);
    }
  }

  // 2. 다른 프로세스의 데이터를 메시지 큐에서 수신
  for (int i = 0; i < total_data_size / num_procs; i += msg_size) {
    long mtype = total_data_size / num_procs * sm_id + i + 1;
    if (msgrcv(msg_queue_id, &msg, sizeof(int) * msg_size, mtype, 0) == -1) {
      perror("msgrcv");
      exit(1);
    }

    for (int j = 0; j < msg_size; j++) {
      final_data[i + j] = msg.mtext[j];
    }
  }

  // 3. 최종 데이터 출력
  printf("SM %d final data: ", sm_id);
  for (int i = 0; i < total_data_size / num_procs; i++) {
    printf("%3d ", final_data[i]);
  }
  printf("\n");
  /**

  */
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

  key = ftok(".", 65);
  if (key == -1)
    perror("ftok");

  msgid = msgget(key, 0666 | IPC_CREAT);

  int h_tile_data[partition_size * partition_size * 8] = {0};

  runProcess(h_tile_data, sm_id, tile_size);
  // h_tile_data에 각 sm의 데이터가 모임.
  // h_title_data[y * WIDTH + x]
  processSM(sm_id, h_tile_data, msgid, 8, tile_size);

  // printf("sm %d :", sm_id);
  // for (int i = 0; i < tile_size * tile_size * 2; ++i) {
  //   printf("%3d ", h_tile_data[i]);
  //   // if ((i + 1) % tile_size * tile_size == 0)
  //   //   printf("\n");
  // }
  // printf("\n");

  return 0;
}
