#include "header.h"

void child_run(int server_id) {
  int server_fd, len;
  struct sockaddr_in address;
  socklen_t addrlen = sizeof(address);
  int buffer[BUFFER_SIZE];
  int clients_fd[8];
  int sndbuf, rcvbuf;
  socklen_t optlen = sizeof(sndbuf);
  int i;

  // 2. 서버 소켓 생성
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    err_exit("Socket failed");

  // if (getsockopt(server_fd, SOL_SOCKET, SO_SNDBUF,
  // &sndbuf,
  //                &optlen) < 0)
  //   err_exit("getsockopt");
  // printf("송신 버퍼 크기: %d\n", sndbuf);

  rcvbuf = 262144;
  if (setsockopt(server_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf,
                 sizeof(rcvbuf)) < 0)
    err_exit("setsockopt");
  if (getsockopt(server_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf,
                 &optlen) < 0)
    err_exit("getsockopt");
  printf("수신 버퍼 크기: %d\n", rcvbuf);
  // 3. 소켓 옵션 설정 (포트 재사용)
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
    printf("server%d Accept : %d\n", server_id, i);
  }

  for (i = 0; i < 8; ++i) {
    char filename[100] = {0};

    len = recv(clients_fd[i], buffer, sizeof(buffer), 0);
    if (len < 0) perror("recv");

    createFilename(filename, "server", "_", server_id,
                   buffer[0] / 2048);
    dirCat(filename, server_id, "server");
    writeDataToFile(filename, buffer, len / 4);
    close(clients_fd[i]);
  }
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