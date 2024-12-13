#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
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

void writeDataToFile(const char *filename, int *data, int data_size) {
  // 파일 디스크립터를 사용한 저수준 파일 생성 및 쓰기
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    perror("Error opening file for writing");
    exit(1);
  }

  ssize_t bytes_written = write(fd, data, data_size * sizeof(int));
  if (bytes_written == -1) {
    perror("Error writing data to file");
    close(fd);
    exit(1);
  }

  close(fd);
}

void createFilename(char *filename, int sm_id, int i) {
  int pos = 0;

  // "sm_" 추가
  filename[pos++] = 's';
  filename[pos++] = 'm';
  filename[pos++] = '_';

  // sm_id 추가
  int tmp_sm_id = sm_id;
  char sm_id_str[10];
  int sm_id_len = 0;
  while (tmp_sm_id > 0) {
    sm_id_str[sm_id_len++] = (tmp_sm_id % 10) + '0';
    tmp_sm_id /= 10;
  }

  // sm_id를 반대로 추가
  for (int j = sm_id_len - 1; j >= 0; --j) {
    filename[pos++] = sm_id_str[j];
  }

  // "_data_" 추가
  filename[pos++] = '_';
  filename[pos++] = 'd';
  filename[pos++] = 'a';
  filename[pos++] = 't';
  filename[pos++] = 'a';
  filename[pos++] = '_';

  // i 추가
  int tmp_i = i;
  char i_str[10];
  int i_len = 0;
  while (tmp_i > 0) {
    i_str[i_len++] = (tmp_i % 10) + '0';
    tmp_i /= 10;
  }

  // i를 반대로 추가
  for (int j = i_len - 1; j >= 0; --j) {
    filename[pos++] = i_str[j];
  }

  // ".txt" 추가
  filename[pos++] = '.';
  filename[pos++] = 't';
  filename[pos++] = 'x';
  filename[pos++] = 't';

  // Null 문자 추가 (문자열 종료)
  filename[pos] = '\0';
}

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

    char filename[50];
    createFilename(filename, sm_id, i);
    writeDataToFile(filename, h_tile_data + (i / 8) * TILE_DATA_SIZE,
                    TILE_DATA_SIZE);
  }

  cudaFree(d_tile_data);
}

void processSM(int sm_id, int *sm_data, int msg_queue_id, int num_procs,
               int tile_size) {
  struct mymsgbuf msg;
  int total_data_size = WIDTH * WIDTH;
  int final_data[total_data_size / num_procs];

  int msg_size = WIDTH / tile_size;

  // 데이터 파일에 저장할 경로
  char filename[50];

  for (int i = 0; i < total_data_size / num_procs; i += msg_size) {
    msg.mtype = sm_data[i] + 1;
    for (int j = 0; j < msg_size; j++)
      msg.mtext[j] = sm_data[i + j];

    if (msgsnd(msg_queue_id, &msg, sizeof(int) * msg_size, 0) == -1) {
      perror("msgsnd");
      exit(1);
    }

    // i에 대한 다른 처리
    createFilename(filename, sm_id, i); // 'i' 값으로 filename을 갱신
  }

  // 2. 다른 프로세스의 데이터를 메시지 큐에서 수신
  for (int i = 0; i < total_data_size / num_procs; i += msg_size) {
    long mtype = total_data_size / num_procs * sm_id + i + 1;

    if (msgrcv(msg_queue_id, &msg, sizeof(int) * msg_size, mtype, 0) == -1) {
      perror("msgrcv");
      exit(1);
    }

    for (int j = 0; j < msg_size; j++)
      final_data[i + j] = msg.mtext[j];
  }

  // 데이터 파일에 기록
  createFilename(filename, sm_id,
                 0); // 'i' 값은 0부터 시작하거나 적절한 값으로 설정
  writeDataToFile(filename, final_data, total_data_size / num_procs);

  // 3. 최종 데이터 출력
  printf("SM %d final data: ", sm_id);
  for (int i = 0; i < total_data_size / num_procs; i++)
    printf("%3d ", final_data[i]);
  printf("\n");
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

  key = ftok(".", 65);
  if (key == -1)
    perror("ftok");

  msgid = msgget(key, 0666 | IPC_CREAT);

  runProcess(h_tile_data, sm_id, tile_size);
  // h_tile_data에 각 sm의 데이터가 모임.
  processSM(sm_id, h_tile_data, msgid, 8, tile_size);

  return 0;
}
