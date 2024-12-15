#include "header.h"

void processSM(int sm_id, int *sm_data, int msg_queue_id,
               int num_procs, int tile_size,
               int *final_data) {
  struct mymsgbuf msg;
  struct timeval start, end;
  long seconds, microseconds;
  double ms;
  int total_data_size = WIDTH * WIDTH;
  int msg_size = WIDTH / tile_size;
  char filename[100] = {0};
  long type_id = total_data_size / num_procs * sm_id;

  gettimeofday(&start, NULL);
  for (int i = 0; i < total_data_size / num_procs;
       i += msg_size) {
    msg.mtype = sm_data[i] + 1;

    if (msg.mtype == type_id + i + 1)
      memcpy((int *)(final_data + i), (int *)(sm_data + i),
             sizeof(int) * msg_size);
    else {
      memcpy((int *)msg.mtext, (int *)(sm_data + i),
             sizeof(int) * msg_size);

      if (msgsnd(msg_queue_id, &msg, sizeof(int) * msg_size,
                 0) == -1)
        perror("msgrcv");
    }
  }

  // 2. 다른 프로세스의 데이터를 메시지 큐에서 수신
  for (int i = 0; i < total_data_size / num_procs;
       i += msg_size) {
    if (final_data[i] == -1) {
      if (msgrcv(msg_queue_id, &msg, sizeof(int) * msg_size,
                 type_id + i + 1, 0) == -1)
        err_exit("msgrcv");
      memcpy((int *)(final_data + i), (int *)(msg.mtext),
             sizeof(int) * msg_size);
    }
  }
  gettimeofday(&end, NULL);

  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  ms = (seconds * 1000) + (microseconds / 1000.0);
  printf("sm%d Client-Client Communication : %f ms\n",
         sm_id, ms);

  // 데이터 파일에 기록
  createFilename(filename, "sm", "_sequence_data", sm_id,
                 -1);  // 'i' 값은 0부터 시작하거나 적절한
                       // 값으로 설정
  dirCat(filename, sm_id, "sm");
  writeDataToFile(filename, final_data,
                  total_data_size / num_procs);
}