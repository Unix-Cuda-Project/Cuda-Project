#include "header.h"

void recv_all(char *buffer, int fd) {
  int len_sum = 0, len = 0;

  int io_size = 1024 * sizeof(int);

  while (len_sum < io_size) {
    len = recv(fd, buffer + len_sum, io_size - len, 0);
    if (len < 0) {
      printf("%d\n", len_sum);
      err_exit("recv");
    }
    len_sum += len;
  }
}

void child_run(int server_id) {
  int server_fd, i;
  struct sockaddr_in address;
  socklen_t addrlen = sizeof(address);
  char buffer[BUFFER_SIZE * sizeof(int)];
  int clients_fd[8];

  struct timeval start, end;
  long seconds, microseconds;
  double ms;

  // 2. 서버 소켓 생성
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    err_exit("Socket failed");

  int opt = 1;
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt,
                 sizeof(opt)) < 0)
    err_exit("Setsockopt failed");

  // 4. 서버 주소 설정
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(PORT + server_id);

  // 5. 소켓 바인딩
  if (bind(server_fd, (struct sockaddr *)&address,
           sizeof(address)) < 0)
    err_exit("Bind failed");

  // 6. 서버 리슨
  if (listen(server_fd, 8) < 0) err_exit("Listen failed");

  for (i = 0; i < 8; ++i) {
    clients_fd[i] = accept(
        server_fd, (struct sockaddr *)&address, &addrlen);
  }

  gettimeofday(&start, NULL);
  for (i = 0; i < 8; ++i) {
    char filename[100] = {0};
    int first_val;

    recv_all(buffer, clients_fd[i]);

    memcpy(&first_val, buffer, 4);
    createFilename(filename, "server", "_", server_id,
                   first_val / 2048);
    dirCat(filename, server_id, "server");
    writeDataToFile(filename, (int *)buffer, 1024);
    close(clients_fd[i]);
  }
  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  ms = (seconds * 1000) + (microseconds / 1000.0);

  printf("Server%d ServerIO : %f ms\n", server_id, ms);
}

int main() {
  int i;
  for (i = 0; i < 2; ++i) {
    switch (fork()) {
      case -1:
        perror("fork");
        exit(1);
        break;
      case 0:
        child_run(i);
        exit(0);
    }
  }

  for (int i = 0; i < 2; ++i) wait(NULL);
  return 0;
}