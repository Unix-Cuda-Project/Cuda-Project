#include "header.h"

void processSM(int sm_id, int *sm_data, int msg_queue_id,
               int num_procs, int tile_size,
               int *final_data) {
  struct mymsgbuf msg;
  int total_data_size = WIDTH * WIDTH;

  int msg_size = WIDTH / tile_size;

  // 데이터 파일에 저장할 경로
  char filename[100] = {0};

  for (int i = 0; i < total_data_size / num_procs;
       i += msg_size) {
    msg.mtype = sm_data[i] + 1;
    for (int j = 0; j < msg_size; j++)
      msg.mtext[j] = sm_data[i + j];

    if (msgsnd(msg_queue_id, &msg, sizeof(int) * msg_size,
               0) == -1)
      perror("msgrcv");
    if (msgrcv(msg_queue_id, &msg, sizeof(int) * msg_size,
               msg.mtype, IPC_NOWAIT) > 0) {
      for (int j = 0; j < msg_size; j++)
        final_data[i + j] = msg.mtext[j];
    }
  }

  // 2. 다른 프로세스의 데이터를 메시지 큐에서 수신
  for (int i = 0; i < total_data_size / num_procs;
       i += msg_size) {
    long mtype =
        total_data_size / num_procs * sm_id + i + 1;

    if (final_data[i] == -1) {
      if (msgrcv(msg_queue_id, &msg, sizeof(int) * msg_size,
                 mtype, 0) == -1)
        err_exit("msgrcv");
      for (int j = 0; j < msg_size; j++)
        final_data[i + j] = msg.mtext[j];
    }
  }

  // 데이터 파일에 기록
  createFilename(
      filename, "sm", "_squence_data", sm_id,
      -1);  // 'i' 값은 0부터 시작하거나 적절한 값으로 설정
  dirCat(filename, sm_id, "sm");
  writeDataToFile(filename, final_data,
                  total_data_size / num_procs);

  // 3. 최종 데이터 출력
  //   printf("SM %d final data: ", sm_id);
  //   for (int i = 0; i < total_data_size / num_procs; i++)
  //     printf("%3d ", final_data[i]);
  //   printf("\n");
  // }
}